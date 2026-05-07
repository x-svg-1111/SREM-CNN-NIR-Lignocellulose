#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srem_nir.data.synthetic import save_synthetic_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data_demo" / "synthetic_srem_demo.npz"))
    ap.add_argument("--n-samples", type=int, default=550)
    ap.add_argument("--n-wavenumbers", type=int, default=3001)
    args = ap.parse_args()
    path = save_synthetic_npz(args.out, n_samples=args.n_samples, n_wavenumbers=args.n_wavenumbers)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
