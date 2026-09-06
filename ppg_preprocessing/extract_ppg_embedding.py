import pandas as pd
import numpy as np
import os
import sys
import torch
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from torch_ecg._preprocessors import Normalize

# 1. Add PaPaGei repo to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths
paths.add_papagei_to_path()

from preprocessing.ppg import preprocess_one_ppg_signal
from segmentations import waveform_to_segments
from linearprobing.utils import resample_batch_signal, load_model_without_module_prefix
from models.resnet import ResNet1DMoE

norm = Normalize(method='z-score')

# print the original (without signal) length

# --- Configuration ---
INPUT_DIR = str(paths.data_root() / "ppg_segments_combined.parquet")
OUTPUT_DIR = str(paths.data_root() / "ppg_embeddings_final.parquet")
# The PaPaGei-s checkpoint. Filename as distributed, parentheses and all.
MODEL_PATH = str(paths.ckpt_root() / "papagei_s (1).pt")
TARGET_FS = 125.0        
WIN_S = 10               

# PaPaGei-S Model Configuration
MODEL_CONFIG = {
    'base_filters': 32,
    'kernel_size': 3,
    'stride': 2,
    'groups': 1,
    'n_block': 18,
    'n_classes': 512, # Embedding dimension
    'n_experts': 3
}

def discard_record(seg_array, atol=1e-6):
    """Filter out flatlines or constant signals"""
    if seg_array.size == 0: return True
    valid_data = seg_array[np.isfinite(seg_array)]
    if valid_data.size == 0: return True
    is_constant = (np.nanmax(valid_data) - np.nanmin(valid_data) <= atol)
    is_all_zeros = np.all(np.abs(valid_data) <= atol)
    return is_constant or is_all_zeros

def load_papagei_model(model_path, device):
    """Initialize and load the pre-trained PaPaGei-S model"""
    model = ResNet1DMoE(
        in_channels=1,
        base_filters=MODEL_CONFIG['base_filters'],
        kernel_size=MODEL_CONFIG['kernel_size'],
        stride=MODEL_CONFIG['stride'],
        groups=MODEL_CONFIG['groups'],
        n_block=MODEL_CONFIG['n_block'],
        n_classes=MODEL_CONFIG['n_classes'],
        n_experts=MODEL_CONFIG['n_experts']
    )
    model = load_model_without_module_prefix(model, model_path)
    model.to(device)
    model.eval()

    print("Model loaded on device: ", device)
    return model

if __name__ == "__main__":
    # print the original (without signal) length

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_papagei_model(MODEL_PATH, device)

    # Load Data
    print(f"Loading raw segments from {INPUT_DIR}...")
    df = pd.read_parquet(INPUT_DIR)
    unique_patients = len(set(df["patient_id"]))
    print(f"unique patients in the original PPG dataset: {unique_patients}")
    final_results = []
    FLUSH_EVERY = 50  # write to parquet every N input rows
    parquet_writer = None

    # Counters for pipeline statistics
    n_total = 0
    n_parse_error = 0
    n_discarded_raw = 0
    n_missing_time = 0
    n_discarded_after_norm = 0
    n_empty_after_denoise = 0
    n_no_segments = 0
    n_empty_after_resample = 0
    n_bad_shape = 0
    n_model_error = 0
    n_success = 0

    # iterate over each PPG SEGMENT
    for row in tqdm(df.itertuples(), total=len(df), desc="Extracting Embeddings"):
        n_total += 1

        # Parse signal (it's stored as JSON string in your raw parquet)
        try:
            raw_sig = np.array(json.loads(row.signal))
        except:
            print(f"Error parsing signal for record {row.record_id}")
            n_parse_error += 1
            continue

        if discard_record(raw_sig):
            n_discarded_raw += 1
            continue

        if row.start_time is None:
            print(f"ERROR: Missing start_time for record {row.record_id}, skipping.")
            n_missing_time += 1
            continue

        try:
            # Step 0: Z-score normalize (as in official notebook)
            raw_sig, _ = norm.apply(raw_sig, fs=row.fs)
            if discard_record(raw_sig):
                print(f"  [{row.record_id}] Discarded after normalization (flatline/zero).")
                n_discarded_after_norm += 1
                continue
            if row.Index % 1000 == 0:
                preview_sig = np.array2string(raw_sig[:5], precision=4, separator=', ')
                print(f"\nProcessing record {row.record_id} (patient {row.patient_id}, segment {row.segment}):")
                print(f"  Normalized signal preview: {preview_sig} ...")

            # Step 1: PaPaGei Denoise
            signal_processed, _, _, _ = preprocess_one_ppg_signal(waveform=raw_sig, frequency=row.fs)
            if signal_processed is None or len(signal_processed) == 0:
                print(f"  [{row.record_id}] Empty after denoising, skipping.")
                n_empty_after_denoise += 1
                continue

            # Step 2: Segmentation (Original FS)
            seg_len_samples = int(row.fs * WIN_S)
            segmented_signals = waveform_to_segments(
                waveform_name='ppg',
                segment_length=seg_len_samples,
                clean_signal=signal_processed
            )
            if len(segmented_signals) == 0:
                print(f"  [{row.record_id}] No complete 10s segments after segmentation (signal too short), skipping.")
                n_no_segments += 1
                continue

            # Step 3: Resample to 125Hz
            resampled_batch = resample_batch_signal(
                segmented_signals,
                fs_original=row.fs,
                fs_target=TARGET_FS,
                axis=-1
            )
            if resampled_batch is None or np.array(resampled_batch).size == 0:
                print(f"  [{row.record_id}] Empty after resampling, skipping.")
                n_empty_after_resample += 1
                continue

            # Step 4: Extract Embeddings via Model
            # Shape should be [num_segments, 1, 1250]
            signal_tensor = torch.Tensor(resampled_batch).unsqueeze(dim=1).to(device)
            expected_len = int(TARGET_FS * WIN_S)  # 1250
            if signal_tensor.shape[1] != 1 or signal_tensor.shape[2] != expected_len:
                print(f"  [{row.record_id}] Unexpected tensor shape {tuple(signal_tensor.shape)}, expected [N, 1, {expected_len}], skipping.")
                n_bad_shape += 1
                continue

            with torch.inference_mode():
                outputs = model(signal_tensor)
                # outputs[0] is the embeddings [num_segments, 512]
                embeddings = outputs[0].cpu().detach().numpy()

            # Step 5: Save results
            for i, emb in enumerate(embeddings):
                final_results.append({
                    'patient_id': row.patient_id,
                    'record_id': row.record_id,
                    'segment_id': row.segment,
                    'window_index': i,
                    'start_time': row.start_time + pd.to_timedelta(i * WIN_S, unit='s'),
                    'embedding': emb.astype(np.float32).tolist(), # Store 512-dim vector
                    'signal': json.dumps(resampled_batch[i].tolist(), separators=(",", ":")),
                    'fs': TARGET_FS,
                })
                if i == 0 and row.Index % 1000 == 0:
                    emb_preview = np.array2string(emb[:5], precision=4, separator=', ')
                    print(f"  [{row.record_id}] window {i}: {emb_preview} ... (512-dim)")
                    print("--------------------------------")
            n_success += 1

        except Exception as e:
            import traceback
            print(f"Error in record {row.record_id}: {e}")
            traceback.print_exc()
            print(f"Error in record {row.record_id}: {e}")
            n_model_error += 1

        # Flush accumulated results to parquet every FLUSH_EVERY input rows
        if n_total % FLUSH_EVERY == 0 and final_results:
            batch_df = pd.DataFrame(final_results)
            batch_table = pa.Table.from_pandas(batch_df, preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(OUTPUT_DIR, batch_table.schema)
            parquet_writer.write_table(batch_table)
            final_results.clear()

    # Flush any remaining results
    if final_results:
        batch_df = pd.DataFrame(final_results)
        batch_table = pa.Table.from_pandas(batch_df, preserve_index=False)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(OUTPUT_DIR, batch_table.schema)
        parquet_writer.write_table(batch_table)
        final_results.clear()

    if parquet_writer is not None:
        parquet_writer.close()

    print(f"\n{'='*50}")
    print(f"Pipeline Summary ({n_total} total segments):")
    print(f"  Parse error:              {n_parse_error:6d}")
    print(f"  Discarded (raw flatline): {n_discarded_raw:6d}")
    print(f"  Missing start_time:       {n_missing_time:6d}")
    print(f"  Discarded (post-norm):    {n_discarded_after_norm:6d}")
    print(f"  Empty after denoising:    {n_empty_after_denoise:6d}")
    print(f"  No 10s segments:          {n_no_segments:6d}")
    print(f"  Empty after resample:     {n_empty_after_resample:6d}")
    print(f"  Bad tensor shape:         {n_bad_shape:6d}")
    print(f"  Model/other error:        {n_model_error:6d}")
    print(f"  Successfully processed:   {n_success:6d}")
    print(f"{'='*50}")

    if parquet_writer is not None:
        print(f"\nSuccess! Embeddings saved incrementally to {OUTPUT_DIR}")
    else:
        print("\nNo embeddings extracted.")
        