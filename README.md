# Tracking date palm loss and gain in Pléiades time series (Qatar)

Pilot code for detecting date palm loss/gain events from multi-temporal
Pléiades imagery of 36 farms in Qatar (2012-2026). It covers data audit,
registration, site/observation matrices, event candidates, a semi-synthetic
benchmark (gate G1) and evaluation on human-labelled real events (gate G2).

Manuscript: not available yet.

## Data availability

The Pléiades imagery is **not redistributed**: it is under a private licence
from the Ministry of Municipality (Qatar). Farm identifiers are anonymised
(`F001` ... `F037`). To run on your own data, lay it out as
`<DATA_ROOT>/<SERVICE>/<FARM>/<FARM>_<SERVICE>{.tif,_mask.tif,...}`
using the same anonymised folder names, and set `DATA_ROOT`
(default `data/datasets`). Intermediate outputs go to `pilot/out/` (git-ignored).

## Layout

- `pilot/gates.yaml` - go/no-go gate thresholds (G0-G2).
- `pilot/*.py` - pipeline stages, run in this order:
  `s0_audit` -> `s1_registration` -> (S2/S3 detection: `notebooks/01_palmseg_annotation_colab.ipynb`)
  -> `s4_sites` -> `s5_events` -> `s5b_filmstrip` (blind human review package)
  -> `s6_synth` -> `s7_g1` -> `s8_g2`.
  Helpers: `g2_labels`, `g2_rereview`, `explore_g2`, `explore_plots`, `paper_figs`, `paper_pdf`.
- `pilot/notebooks/` - Colab notebooks (annotation, G2 exploration).
- `pilot/tests/` - pytest suite.

## Tests

```
pip install numpy pandas scipy scikit-learn rasterio pyyaml pyarrow matplotlib pytest
python -m pytest pilot/tests
```

## License

MIT, code only (see `LICENSE`).
