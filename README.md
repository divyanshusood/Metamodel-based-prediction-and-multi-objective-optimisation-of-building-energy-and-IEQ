# Building Energy Analysis and Optimisation

This repository contains code and workflows for analyzing building energy performance, occupant comfort, and indoor environmental quality.  
Originally developed in Jupyter notebooks, the analysis has been consolidated into a single Python script for easier execution and version control.

---

## Features

- Data preprocessing and cleaning (outlier detection, filtering, column renaming).
- Principal Component Analysis (PCA) for dimensionality reduction.
- Sobol sensitivity analysis for input–output relationships.
- Predictive modelling using **XGBoost** and **scikit-learn** regressors.
- Evaluation of heating, thermal discomfort hours, and CO₂ concentration.
- Optimisation routines to balance energy and comfort trade-offs.
- Plotting and visualisation (all results generated at the end of the script).

---

## Repository Structure

```
.
├─ combined_analysis.py      # Main analysis script (runs end-to-end)
├─ notebooks/                # Original sanitized Jupyter notebooks
│  ├─ Main_0.ipynb
│  ├─ Main_1.ipynb
│  └─ Main_3_plotting.ipynb
├─ requirements.txt          # Python dependencies
├─ .gitignore
├─ LICENSE                   # MIT License
└─ publish_to_github.sh      # Helper script to push repo to GitHub
```

---

## Getting Started

### 1. Clone or download the repository
```bash
git clone https://github.com/<USERNAME>/<REPO>.git
cd <REPO>
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the analysis
```bash
python combined_analysis.py
```

> Replace `<PATH_PLACEHOLDER>` with the correct local path to your dataset (e.g., CSV file).  
> Replace `<SECRET_PLACEHOLDER>` with any required API keys (if applicable).  

The script will run the entire workflow and produce results and plots at the end.

---

## Dependencies

Detected from the notebooks and included in `requirements.txt`:

- numpy  
- pandas  
- scipy  
- scikit-learn  
- xgboost  
- seaborn  
- matplotlib  
- statsmodels  
- SALib  
- bayes_opt  
- deap  

---

## Reproducibility Notes

- Keep raw data files in a `data/` folder (this is already ignored by `.gitignore`).
- Do not commit API keys or credentials — use environment variables or `.env` files.
- For large projects, consider splitting the workflow into modules (`data.py`, `models.py`, `plots.py`).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
