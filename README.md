# Research Notebooks: Data Analysis & Plotting

This repository contains sanitized Jupyter notebooks for data processing, analysis, and plotting.
All outputs and absolute file paths have been removed to make the code portable and GitHub-friendly.
Replace placeholders like `<PATH_PLACEHOLDER>` or `<SECRET_PLACEHOLDER>` with your own values when running locally.

## Notebook Topics

- Sobol
- PCA
- machine learning prediction model for heating, thermal discomfort HRS and co2 concentration > 1000 ppm HRS
- comparison separately
- all three outputs combined
- multi-objective optimisation
- validation

## Getting Started

1. **Create and activate a virtual environment** (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Open the notebooks**:

```bash
jupyter notebook notebooks/
```

> If the notebooks reference files, update any `<PATH_PLACEHOLDER>` values to point to your local data. Keep large datasets out of Git by putting them under the `data/` folder (already ignored via `.gitignore`).

## Repository Structure

```
.
├─ notebooks/
│  ├─ Main_0.ipynb
│  ├─ Main_1.ipynb
│  └─ Main_3_plotting.ipynb
├─ requirements.txt
├─ .gitignore
├─ LICENSE
└─ publish_to_github.sh
```

## Reproducibility Tips

- Keep raw data in `data/` (ignored by Git).
- Add configuration values via environment variables or a `.env` file (also ignored).
- Avoid hardcoding file paths; use `pathlib.Path` and project-relative paths.

