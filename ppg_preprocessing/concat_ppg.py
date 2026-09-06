"""Chain temporally adjacent PPG segments into continuous runs.

Stage 2 for both cohorts, but they land in different shapes, and the asymmetry
is real rather than an oversight:

    IV   ppg_segments_raw_IV.parquet  --per-patient-->  MIMIC_IV_ppg_by_patient/<pid>/ppg.parquet
    III  ppg_segments_raw_III_<g>.parquet  --shard-->   MIMIC_III_ppg_combined/ppg_segments_combined_III_<g>.parquet
                                                        then split by ppg_parquet_per_patient.py

MIMIC-III is sharded by patient group (p00..p09) all the way from extraction,
so it stays sharded here and is split into per-patient files one stage later.
MIMIC-IV is one file, so it goes straight to per-patient. Collapsing the two
into one shape would leave ppg_parquet_per_patient.py with no input.

WHAT CHAINING DOES, AND HOW TO TELL IT RAN
------------------------------------------
Adjacent segments of the same patient/record/channel whose timestamps meet
within one sample are concatenated into a single row, up to MAX_DURATION_S.
The `segment` field of a chained row is the joined names of its parts --
"3238451_0002+3238451_0003+..." -- so a `+` in that column is proof this stage
ran, and its absence on a multi-segment record is proof it did not.

THE III ENTRY POINT IS RECONSTRUCTED, NOT RECOVERED
---------------------------------------------------
The research code had only the IV branch below; the III shards were produced
by editing its paths by hand between runs, and that edit was never committed.
`--cohort III` reconstructs that step from the surviving evidence: the output
filenames ppg_parquet_per_patient.py reads, and the chained `+` names in the
files it produced. It calls the SAME combine_continuous_segments the IV branch
calls, so the transformation is identical -- only the output shape differs.

It has not been checked against the original III shards, which no longer
exist. Anyone reproducing MIMIC-III from raw waveforms should treat this stage
as the one to verify first.

Usage:
    python concat_ppg.py --cohort IV  --raw <raw_IV.parquet>
    python concat_ppg.py --cohort III --raw <raw_III_p00.parquet> --group p00
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import gc

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import paths

MAX_DURATION_S = 3600  # 1 hour

def combine_continuous_segments(df):
    """
    Combine temporally adjacent segments from the same patient/record/channel
    where the previous segment's end_time matches the next segment's start_time.
    """
    # Sort to ensure correct temporal order
    df = df.sort_values(
        ["patient_id", "record_id", "channel_name", "start_time"]
    ).reset_index(drop=True)

    combined = []
    i = 0

    while i < len(df):
        row = df.iloc[i]
        
        # Start a chain: accumulate signal from this row
        sig = np.array(json.loads(row.signal))
        chain_start = row.start_time
        chain_end = row.end_time_exclusive
        chain_fs = row.fs
        chain_patient = row.patient_id
        chain_record = row.record_id
        chain_channel = row.channel_name
        seg_names = [row.segment]


        # Try to extend the chain with subsequent rows
        j = i + 1
        while j < len(df):
            if len(sig) / chain_fs >= MAX_DURATION_S:
                break
            nxt = df.iloc[j]

            # Must be same patient, record, channel, fs
            if (nxt.patient_id != chain_patient
                or nxt.record_id != chain_record
                or nxt.channel_name != chain_channel
                or not np.isclose(nxt.fs, chain_fs, rtol=0, atol=1e-8)):
                break

            # Check temporal continuity (tolerance < 1 sample)
            if chain_end is None or nxt.start_time is None:
                break

            gap = abs((nxt.start_time - chain_end).total_seconds())
            if gap > (1.0 / chain_fs):
                break

            # Extend the chain
            nxt_sig = np.array(json.loads(nxt.signal))
            sig = np.concatenate([sig, nxt_sig])
            chain_end = nxt.end_time_exclusive
            seg_names.append(nxt.segment)
            j += 1

        combined.append({
            "patient_id": chain_patient,
            "record_id": chain_record,
            "segment": "+".join(seg_names),  # e.g. "0003+0004+0005"
            "channel_name": chain_channel,
            "fs": chain_fs,
            "n_samples": len(sig),
            "start_time": chain_start,
            "end_time_exclusive": chain_end,
            "signal": json.dumps(sig.tolist(), separators=(",", ":")),
        })

        if np.random.random() < 1/30:
            print(f"  patient={chain_patient} | segments={'+'.join(seg_names)} | duration={len(sig)/chain_fs:.1f}s | start={chain_start}")

        i = j  # skip all rows we consumed

    df_out = pd.DataFrame(combined)
    print(f"  {len(df)} → {len(df_out)} segments (reduced by {len(df) - len(df_out)})")
    return df_out


def _chain_per_patient(df_all, out_root):
    """MIMIC-IV shape: one directory per patient, one ppg.parquet inside it.

    Patient by patient rather than all at once -- a whole cohort of decoded
    signal does not fit in memory, and an interrupted run resumes because
    finished patients are skipped.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for pid, df_pid in tqdm(df_all.groupby("patient_id", sort=False),
                            desc="Patients"):
        pid_str = str(pid)
        patient_dir = out_root / pid_str
        patient_dir.mkdir(parents=True, exist_ok=True)
        out_path = patient_dir / "ppg.parquet"

        if out_path.exists():
            print(f"Skipping existing patient {pid_str}", flush=True)
            continue

        print(f"\nProcessing patient {pid_str}", flush=True)
        print(f"  raw rows: {len(df_pid)}", flush=True)

        df_combined = combine_continuous_segments(df_pid)
        df_combined.to_parquet(str(out_path), index=False)
        print(f"  Saved to {out_path}", flush=True)

        n_done += 1
        del df_combined
        gc.collect()
    return n_done


def _chain_to_shard(df_all, out_path):
    """MIMIC-III shape: one combined shard mirroring the raw shard.

    Chaining is still done per patient -- combine_continuous_segments sorts by
    patient anyway, so grouping first changes nothing about the result and
    keeps peak memory to one patient. Rows are appended through a single
    ParquetWriter so the shard is written once, in raw-file order.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n_rows = 0
    try:
        for pid, df_pid in tqdm(df_all.groupby("patient_id", sort=False),
                                desc="Patients"):
            df_combined = combine_continuous_segments(df_pid)
            if len(df_combined) == 0:
                continue
            table = pa.Table.from_pandas(df_combined, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema,
                                          compression="snappy")
            writer.write_table(table)
            n_rows += len(df_combined)
            del df_combined, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    return n_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', choices=['III', 'IV'], required=True)
    ap.add_argument('--raw', type=Path, default=None,
                    help="raw parquet from the extract stage. Default: "
                         "$PPG_DATA_ROOT/ppg_segments_raw_IV.parquet for IV, "
                         "$PPG_DATA_ROOT/ppg_segments_raw_III_<group>.parquet "
                         "for III.")
    ap.add_argument('--group', default=None, metavar='GROUP',
                    help="III only: patient group, e.g. p00. Required for III "
                         "because both the input and the output are sharded "
                         "by group.")
    ap.add_argument('--out', type=Path, default=None,
                    help="IV: the per-patient root, default "
                         "$PPG_DATA_ROOT/MIMIC_IV_ppg_by_patient. "
                         "III: the combined shard, default $PPG_DATA_ROOT/"
                         "MIMIC_III_ppg_combined/ppg_segments_combined_III_<group>.parquet")
    args = ap.parse_args()

    if args.cohort == 'III' and not args.group and not (args.raw and args.out):
        ap.error("--cohort III needs --group (or both --raw and --out), "
                 "because III is sharded by patient group on both sides.")

    if args.cohort == 'IV':
        raw_path = args.raw or (paths.data_root() / "ppg_segments_raw_IV.parquet")
        out = args.out or (paths.data_root() / "MIMIC_IV_ppg_by_patient")
    else:
        raw_path = args.raw or (
            paths.data_root() / f"ppg_segments_raw_III_{args.group}.parquet")
        out = args.out or (
            paths.data_root() / "MIMIC_III_ppg_combined"
            / f"ppg_segments_combined_III_{args.group}.parquet")

    if not raw_path.exists():
        raise SystemExit(f"raw input not found: {raw_path}")

    print(f"Loading full parquet from {raw_path}", flush=True)
    df_all = pd.read_parquet(str(raw_path))
    print(f"Loaded {len(df_all)} rows, "
          f"{df_all['patient_id'].nunique()} patients", flush=True)

    if args.cohort == 'IV':
        n = _chain_per_patient(df_all, out)
        print(f"\nDone. {n} patient directories written under {out}")
    else:
        n = _chain_to_shard(df_all, out)
        print(f"\nDone. {n} chained rows written to {out}")

    del df_all
    gc.collect()
