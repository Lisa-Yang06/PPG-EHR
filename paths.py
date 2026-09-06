"""Every path that points OUTSIDE this repository, resolved in one place.

WHY THIS FILE EXISTS
--------------------
The research version of this code carried 164 distinct absolute paths, all
descending from one researcher's cluster account. None of them exist on anyone
else's machine, so nothing ran anywhere else. Every one of those literals is
now expressed as a root from the environment plus a relative sub-path, and the
roots are declared here so there is exactly one place to look.

THE FOUR ROOTS
--------------
    PPG_DATA_ROOT     derived data: patient_history_db_*.joblib, the per-patient
                      PPG parquet trees, the split files, the ds*/ task dirs.
                      Everything the pipeline writes and re-reads.

    PPG_CKPT_ROOT     model weights: best_medbert_model_*.pt (sEHR-BERT MLM),
                      best_aligned_model_*.pt (contrastive alignment),
                      the PaPaGei-s checkpoint.

    PAPAGEI_REPO      a clone of the PaPaGei foundation-model repository. It is
                      imported, not vendored, because it has its own licence and
                      its own release history. See README for the pinned commit.

    PPG_CACHE_ROOT    optional. Only meaningful on a cluster whose home
                      directory is small or read-only; see configure_caches().

MIMIC ITSELF IS NOT A ROOT HERE
-------------------------------
The raw MIMIC-III/IV waveform and CSV trees are read only by the first
preprocessing stage, which takes them as explicit arguments. They are
credentialed data under a PhysioNet DUA and are deliberately not wired into a
global default, so no script can silently reach for them.

WHY ENVIRONMENT VARIABLES AND NOT A CONFIG FILE
-----------------------------------------------
A config file has to be found before it can be read, which reintroduces the
problem it is meant to solve. An environment variable is already inherited by
SLURM jobs, needs no parser, and cannot be accidentally committed. Copy
.env.example to .env and source it, or export the four variables in a job
script -- both work, and neither is checked in.

Roots resolve LAZILY, on first use. Importing this module never fails, so
`--help` and unit tests keep working on a machine with none of the data.
"""
import os
import sys
from pathlib import Path

# What each root is for, quoted back at the user when one is missing. A stack
# trace saying KeyError: 'PPG_DATA_ROOT' tells someone nothing about what to
# set it to.
_ROOT_DOC = {
    'PPG_DATA_ROOT': 'derived data (patient_history_db_*.joblib, the PPG '
                     'parquet trees, splits/, the ds*/ task directories)',
    'PPG_CKPT_ROOT': 'model checkpoints (best_medbert_model_*.pt, '
                     'best_aligned_model_*.pt, the PaPaGei-s weights)',
    'PAPAGEI_REPO': 'a clone of the PaPaGei foundation-model repository',
    'PPG_CACHE_ROOT': 'a writable cache directory (optional)',
}

_cache = {}


def _root(var, required=True):
    if var in _cache:
        return _cache[var]
    raw = os.environ.get(var)
    if not raw:
        if not required:
            return None
        raise RuntimeError(
            f"{var} is not set.\n"
            f"  It should point to {_ROOT_DOC[var]}.\n"
            f"  Copy .env.example to .env and edit it, or export {var} "
            f"in your job script."
        )
    p = Path(raw).expanduser()
    if required and not p.exists():
        raise RuntimeError(
            f"{var} is set to {p}, which does not exist.\n"
            f"  It should point to {_ROOT_DOC[var]}."
        )
    _cache[var] = p
    return p


def data_root():
    """Derived data. Written and re-read by the pipeline."""
    return _root('PPG_DATA_ROOT')


def ckpt_root():
    """Model checkpoints."""
    return _root('PPG_CKPT_ROOT')


def papagei_repo():
    """The PaPaGei repository clone."""
    return _root('PAPAGEI_REPO')


def cache_root():
    """Optional writable cache root, or None."""
    return _root('PPG_CACHE_ROOT', required=False)


def add_papagei_to_path():
    """Make PaPaGei's modules importable.

    Replaces `sys.path.append("/home/<user>/papagei-foundation-model")`, which
    appeared verbatim in 30 places. PaPaGei is imported rather than vendored,
    so this stays a path insertion rather than becoming a package dependency.

    Call it BEFORE `from preprocessing.ppg import ...`, `from models.resnet
    import ...` or `import linearprobing.utils`.
    """
    p = str(papagei_repo())
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def configure_caches():
    """Opt-in cluster cache redirection. Does NOTHING unless PPG_CACHE_ROOT is set.

    The research code set HOME, XDG_CACHE_HOME and MPLCONFIGDIR unconditionally
    to a scratch directory, because the cluster's real home was too small for
    matplotlib's font cache. Twenty-eight files reassigned HOME outright.

    That is correct on that cluster and actively harmful anywhere else --
    silently repointing HOME changes where every downstream library looks for
    credentials, configuration and caches. So it is opt-in: set PPG_CACHE_ROOT
    only if you actually need it, and note that HOME is deliberately NOT
    reassigned here even then. Only the two cache variables are, which is what
    the original hack was really for.
    """
    root = cache_root()
    if root is None:
        return None
    root = Path(root)
    (root / 'matplotlib').mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('XDG_CACHE_HOME', str(root))
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'matplotlib'))
    return root


def describe():
    """Print what is currently configured. `python paths.py`."""
    print("resolved roots")
    print("-" * 60)
    for var in _ROOT_DOC:
        raw = os.environ.get(var)
        if not raw:
            opt = ' (optional)' if var == 'PPG_CACHE_ROOT' else ''
            print(f"  {var:<16} NOT SET{opt}")
            continue
        p = Path(raw).expanduser()
        print(f"  {var:<16} {p}" + ("" if p.exists() else "   <-- MISSING"))


if __name__ == '__main__':
    describe()
