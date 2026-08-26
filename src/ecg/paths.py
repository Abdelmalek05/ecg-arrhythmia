"""Project paths, resolved from this file so code works from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "physionet"
BUILD = ROOT / "data" / "build"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
REPORTS = ROOT / "reports"
RESULTS_CSV = RESULTS / "results.csv"


def ensure_dirs():
    for d in (RAW, BUILD, RESULTS, FIGURES, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
