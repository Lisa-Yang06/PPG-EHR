"""Extract raw PPG segments from the MIMIC-IV matched waveform database.

Stage 1 of the MIMIC-IV PPG chain:

    THIS FILE       raw waveform -> ppg_segments_raw_IV.parquet
    concat_ppg.py   chain        -> MIMIC_IV_ppg_by_patient/<pid>/ppg.parquet
    ppg_full_preprocessing.py    -> MIMIC_IV_ppg_filtered_by_patient/

Raw means raw: adjacent segments are chained in the next stage, not here.

MIMIC-IV is not sharded by patient group -- one pass writes one file -- which
is why there is no --groups flag here and extract_ppg_III.py has one. Every
record is accumulated in memory before writing, so this stage wants a
high-memory node.

The waveform tree is credentialed PhysioNet data and has no default path;
pass --waveform-dir explicitly.

Usage:
    python extract_ppg_IV.py --waveform-dir /path/to/mimic4wdb/0.1.0/waves
"""
import argparse
import os
import sys
import wfdb
import numpy as np
from pathlib import Path
import re
from datetime import datetime, timedelta
import pandas as pd
import json
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths

PPG_PAT = re.compile(r"(PLETH|PPG|SpO2[ _-]?Pleth)", re.I)


def _combine_dt(date_str, time_str):
    """Combine date and time strings from header into datetime object."""
    if not date_str or not time_str:
        return None
    for fmt in (
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(f"{date_str} {time_str}", fmt)
        except ValueError:
            pass
    return None


def find_ppg_segments(master_hea_path: str):
    """
    Extract raw PPG segments at their native sampling rate and calculate
    absolute timestamps.
    """
    master = Path(master_hea_path)
    rec_dir = master.parent
    record_id = master.stem[:-5]  # Remove '_0000' to get '81739927'
    patient_id = master.parent.parent.name.lstrip("p")

    # Read master header to get baseline time and segment list
    try:
        Hm = wfdb.rdheader(str(master.with_suffix("")).replace("_0000", ""))
    except Exception:
        Hm = wfdb.rdheader(str(master.with_suffix("")))

    base_date = getattr(Hm, "base_date", None)
    base_time = getattr(Hm, "base_time", None)
    base_dt = _combine_dt(base_date, base_time)

    if base_dt is None:
        print(f"WARNING: no base datetime for {master_hea_path} "
            f"(base_date={base_date}, base_time={base_time}), "
            f"timestamps will be None")
        
    seg_names = getattr(Hm, "seg_name", [])
    seg_lens = getattr(Hm, "seg_len", [])
    if seg_names is None or seg_lens is None:
        raise ValueError(f"Not a multi-segment record: {master_hea_path}")
    base_fs = Hm.fs  # base frame rate, e.g. 62.4725
    if not base_fs:
        raise ValueError(f"No base sampling frequency in {master_hea_path}")

    rows = []
    cumulative_seconds = 0.0

    for seg_name, slen in zip(seg_names, seg_lens):
        if seg_name == "~" or slen == 0:
            # Accumulate offset for gaps
            if base_fs:
                cumulative_seconds += slen / base_fs
            continue

        seg_base = str(rec_dir / seg_name)
        try:
            # Check if this segment contains PPG
            Hs = wfdb.rdheader(seg_base)
            ppg_idx = [
                i for i, n in enumerate(Hs.sig_name) if PPG_PAT.search(n or "")
            ]

            if ppg_idx:
                print(f"found PPG in {seg_base}")

                # Read with smooth_frames=False to get native sampling rate
                record = wfdb.rdrecord(
                    seg_base, channels=ppg_idx, smooth_frames=False
                )

                # After channel selection, samps_per_frame is re-indexed
                # e.g. if ppg_idx=[5], record now has 1 channel, index 0
                spf = record.samps_per_frame[0]
                fs = record.fs * spf  # e.g. 62.4725 * 2 = 124.945

                start_time = None
                end_time = None
                if base_dt:
                    start_time = base_dt + timedelta(seconds=cumulative_seconds)

                for j, ch_i in enumerate(ppg_idx):
                    # e_p_signal is a list of 1d arrays, one per channel
                    sig = record.e_p_signal[j]
                    if np.all(np.isnan(sig)):
                        print(f"WARNING: all-NaN signal in {seg_base} ch={Hs.sig_name[ch_i]}, skipping")
                        continue

                    if base_dt:
                        end_time = start_time + timedelta(
                            seconds=len(sig) / fs
                        )

                    rows.append(
                        {
                            "patient_id": patient_id,
                            "record_id": record_id,
                            "segment": seg_name,
                            "seg_num": (
                                seg_name.split("_")[-1]
                                if "_" in seg_name
                                else ""
                            ),
                            "channel_name": Hs.sig_name[ch_i],
                            "fs": float(fs),
                            "n_samples": int(sig.shape[0]),
                            "start_time": start_time,
                            "end_time_exclusive": end_time,
                            "signal": json.dumps(
                                sig.tolist(), separators=(",", ":")
                            ),
                        }
                    )

            # seg_len is always in frames → divide by base frame rate
            cumulative_seconds += slen / base_fs

        except Exception as e:
            print(f"Error processing segment {seg_name} in {master_hea_path}: {e}")
            if base_fs:
                cumulative_seconds += slen / base_fs
            continue

    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--waveform-dir', type=Path, required=True,
                    help="MIMIC-IV matched waveform root (the waves/ "
                         "directory). Credentialed PhysioNet data; "
                         "deliberately has no default.")
    ap.add_argument('--out', type=Path, default=None,
                    help="default: $PPG_DATA_ROOT/ppg_segments_raw_IV.parquet")
    args = ap.parse_args()

    root = args.waveform_dir
    # The cohort is in the output filename, and the path is absolute: the
    # next stage reads this file by that name, and MIMIC-III's shards sit
    # beside it under the same root.
    out_path = args.out or (paths.data_root() / "ppg_segments_raw_IV.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    four_zero_hea_paths = sorted(root.rglob("*_0000.hea"))
    if not four_zero_hea_paths:
        raise SystemExit(
            f"no *_0000.hea master headers under {root}. --waveform-dir "
            f"should be the waves/ directory of the matched subset.")
    print(f"Found {len(four_zero_hea_paths)} master headers", flush=True)

    results = []
    for master in tqdm(four_zero_hea_paths, desc="Extracting Raw PPG"):
        try:
            single_record_result = find_ppg_segments(str(master))
            results.extend(single_record_result)
        except Exception as e:
            print(f"Error in {master}: {e}")

    if not results:
        raise SystemExit("no PPG segments extracted.")

    df = pd.DataFrame(results)
    print(f"Extracted {len(df)} raw PPG segments across "
          f"{df['patient_id'].nunique()} patients.")
    df.to_parquet(str(out_path), index=False)
    print(f"Saved {len(df)} raw segments to {out_path}")

    preview = df.drop(columns=["signal"]).head()
    print("\nFirst 5 rows (signal omitted):")
    print(preview.to_string(index=False))
