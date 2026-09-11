"""Extract raw PPG segments from the MIMIC-III matched waveform database.

Stage 1 of the MIMIC-III PPG chain. Reads .hea/.dat records straight from the
PhysioNet tree and writes one raw parquet shard per patient group (p00..p09).

    THIS FILE            raw waveform  -> ppg_segments_raw_III_<group>.parquet
    concat_ppg.py        combine       -> MIMIC_III_ppg_combined/
    ppg_parquet_per_patient.py  split  -> MIMIC_III_ppg_by_patient/<pid>/ppg.parquet
    ppg_full_preprocessing.py          -> MIMIC_III_ppg_filtered_by_patient/

Raw means raw: segments are NOT joined here. Temporally adjacent segments are
chained by concat_ppg.py in the next stage, which is what puts the "a+b+c"
names in the `segment` column.

The waveform tree is credentialed PhysioNet data and has no default path --
pass --waveform-dir explicitly. Everything this script writes goes under
$PPG_DATA_ROOT.

Usage:
    python extract_ppg_III.py --waveform-dir /path/to/mimic3wdb-matched/1.0
    python extract_ppg_III.py --waveform-dir ... --groups p00 p01   # one job per group
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
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import paths

PPG_PAT = re.compile(r"PLETH", re.I)

master_cache = {}

def find_master_header(patient_dir: Path, record_id: str):
    """Find master header for a specific record."""
    cache_key = str(patient_dir)
    if cache_key not in master_cache:
        masters = {}
        for f in patient_dir.glob("p*-*-*.hea"):
            if f.stem.endswith("n"):
                continue
            try:
                Hm = wfdb.rdheader(str(f.with_suffix("")))
                seg_names = getattr(Hm, "seg_name", []) or []
                for s in seg_names:
                    if s and s != "~":
                        rid = s.rsplit("_", 1)[0]
                        masters[rid] = (f, Hm)
            except Exception:
                continue
        master_cache[cache_key] = masters
    
    return master_cache[str(patient_dir)].get(record_id, (None, None))


def find_ppg_segments_iii(hea_path: str):
    """
    Extract raw PPG segments from MIMIC-III waveform files.
    MIMIC-III has no master header - each segment is a standalone file.
    """
    hea = Path(hea_path)
    patient_dir = hea.parent
    seg_base = str(hea.with_suffix(""))
    patient_id = hea.parent.name.lstrip("p")
    # "<record>_<segment>" -> "<record>"
    record_id = hea.stem.rsplit("_", 1)[0]

    rows = []

    no_start_time_count = 0

    try:
        master, Hm = find_master_header(patient_dir, record_id)
        
        base_dt = None

        seg_names = []
        seg_lens = []
        base_fs = None

        if Hm is not None:
            base_dt = getattr(Hm, "base_datetime", None)
            seg_names = getattr(Hm, "seg_name", []) or []
            seg_lens = getattr(Hm, "seg_len", []) or []
            base_fs = Hm.fs

        if base_dt is None:
            print(f"WARNING: no base datetime for {hea_path}, skipping.")
            no_start_time_count += 1 
            return rows, no_start_time_count

        Hs = wfdb.rdheader(seg_base) 

        start_time = None
        if base_dt and base_fs:
            cumulative_seconds = 0.0
            for sname, slen in zip(seg_names, seg_lens):
                if sname == hea.stem:
                    start_time = base_dt + timedelta(seconds=cumulative_seconds)
                    break
                cumulative_seconds += slen / base_fs

        # Find PPG channels
        ppg_idx = [i for i, n in enumerate(Hs.sig_name) if PPG_PAT.search(n or "")]
        if not ppg_idx:
            return rows, no_start_time_count

        # Read signal
        record = wfdb.rdrecord(seg_base, channels=ppg_idx, smooth_frames=False)

        for j, ch_i in enumerate(ppg_idx):
            spf = record.samps_per_frame[j]
            fs = record.fs * spf

            if record.e_p_signal is not None and spf > 1:
                sig = record.e_p_signal[j]
            else:
                record2 = wfdb.rdrecord(seg_base, channels=[ppg_idx[j]], smooth_frames=True)
                sig = record2.p_signal[:, 0]
                fs = record2.fs

            if np.all(np.isnan(sig)):
                print(f"WARNING: all-NaN signal in {seg_base}, skipping")
                continue
            
            end_time = None

            if start_time is None:
                print(f"WARNING: skipping {hea.stem}, no start_time")
                continue

            segment_duration = Hs.sig_len / Hs.fs
            end_time = start_time + timedelta(seconds=segment_duration)

            rows.append({
                "patient_id": patient_id,
                "record_id": record_id,
                "segment": hea.stem,
                "seg_num": hea.stem.rsplit("_", 1)[-1],
                "channel_name": Hs.sig_name[ch_i],
                "fs": float(fs),
                "n_samples": int(len(sig)),
                "start_time": start_time,
                "end_time_exclusive": end_time,
                "signal": json.dumps(sig.tolist(), separators=(",", ":")),
            })
            if np.random.random() < 1/50:
            # print a preview of signals and full info of other data
                print(f"Extracted {len(sig)} samples at {fs}Hz from {hea_path}")
                print(f"  patient_id: {patient_id}, record_id: {record_id}, channel: {Hs.sig_name[ch_i]}, start_time: {start_time}, end_time: {end_time}")
                print(f"  signal preview: {sig[:10]} ... {sig[-10:]}")
                print("--------------------------------")

    except Exception as e:
        print(f"Error processing {hea_path}: {e}")

    return rows, no_start_time_count


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--waveform-dir', type=Path, required=True,
                    help="MIMIC-III matched waveform root, the directory "
                         "containing p00/ .. p09/. Credentialed PhysioNet "
                         "data; deliberately has no default.")
    ap.add_argument('--groups', nargs='+', default=None, metavar='GROUP',
                    help="patient groups to process, e.g. p00 p03. "
                         "Default: every group found under --waveform-dir. "
                         "Pass one group per job to parallelise on a cluster.")
    ap.add_argument('--out-dir', type=Path, default=None,
                    help="default: $PPG_DATA_ROOT")
    args = ap.parse_args()

    root = args.waveform_dir
    out_dir = args.out_dir or paths.data_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group selection is explicit, and defaults to every group present. A
    # default that covered one group would make this script process a tenth of
    # MIMIC-III and exit 0 -- the failure mode that costs a day to notice.
    all_groups = sorted(p.name for p in root.glob("p0?") if p.is_dir())
    if not all_groups:
        raise SystemExit(
            f"no p0? group directories under {root}. --waveform-dir should be "
            f"the directory that CONTAINS p00/ .. p09/.")
    if args.groups:
        unknown = sorted(set(args.groups) - set(all_groups))
        if unknown:
            raise SystemExit(
                f"group(s) {unknown} not found under {root}. "
                f"available: {all_groups}")
        groups = [g for g in all_groups if g in set(args.groups)]
    else:
        groups = all_groups
    print(f"Processing {len(groups)}/{len(all_groups)} group(s): "
          f"{' '.join(groups)}", flush=True)

    SEG_PAT = re.compile(r"^\d+_\d{4}$")

    no_start_time_total = 0

    for p_dir in [root / g for g in groups]:
        master_cache.clear()

        p_name = p_dir.name  # e.g. "p00"
        print(f"Processing {p_name}...")
        print(f"{'='*50}")

        hea_paths = sorted([
            p for p in p_dir.rglob("*.hea")
            if SEG_PAT.match(p.stem)
        ])
        print(f"Found {len(hea_paths)} .hea files in {p_name}")

        CHUNK_SIZE = 500

        results = []
        total_rows = 0

        output_path = str(out_dir / f"ppg_segments_raw_III_{p_name}.parquet")

        writer = None

        try:
            for hea_path in tqdm(hea_paths, desc=f"Extracting {p_name} PPG"):
                try:
                    rows, cnt = find_ppg_segments_iii(str(hea_path))
                    no_start_time_total += cnt
                    results.extend(rows)
                    del rows

                    if len(results) >= CHUNK_SIZE:
                        df = pd.DataFrame(results)

                        print(
                            f"Writing {len(df)} rows into {output_path}",
                            flush=True
                        )

                        table = pa.Table.from_pandas(df, preserve_index=False)

                        if writer is None:
                            writer = pq.ParquetWriter(
                                output_path,
                                table.schema,
                                compression="snappy"
                            )

                        writer.write_table(table)

                        total_rows += len(df)

                        del df
                        del table
                        results.clear()
                        gc.collect()

                except Exception as e:
                    print(f"Error in {hea_path}: {e}", flush=True)

            if results:
                df = pd.DataFrame(results)

                print(
                    f"Writing final {len(df)} rows into {output_path}",
                    flush=True
                )

                table = pa.Table.from_pandas(df, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        table.schema,
                        compression="snappy"
                    )

                writer.write_table(table)

                total_rows += len(df)

                del df
                del table
                results.clear()
                gc.collect()

        finally:
            if writer is not None:
                writer.close()

        if total_rows > 0:
            print(f"Saved {total_rows} raw PPG rows to {output_path}", flush=True)
        else:
            print(f"No PPG segments found in {p_name}.", flush=True)

    print(f"\nTotal segments skipped due to no base_datetime: {no_start_time_total}")