# Demo data

`synthetic_srem_demo.npz` can be generated locally by:

```bash
python scripts/make_synthetic_data.py
```

It mimics the private data schema:

- `X`: full NIR spectra, default shape `(550, 3001)`
- `y`: cellulose, hemicellulose, and lignin targets, shape `(n_samples, 3)`
- `wavenumbers`: spectral axis in cm^-1, descending from 10000 to 4000
- `sample_ids`: integer identifiers
- `crop_labels`: synthetic crop labels

The values are synthetic and should not be used to reproduce the manuscript
performance table.
