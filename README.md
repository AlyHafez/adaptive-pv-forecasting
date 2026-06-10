# ADAPT-PV: Adaptive Residential Solar PV Forecasting

> **Industry partnership with [Fuse Energy](https://fuseenergy.com)** | MSc Dissertation, UCL 2026

A three-stage adaptive framework for zero-shot residential solar PV forecasting. The TFT foundation model **never sees target household data** — a daily-retrained MLP residual corrector closes the domain gap within 7 days of deployment.

---

## Key Results

| Model | Clean NRMSE | Drifted NRMSE | Shaded NRMSE |
|-------|-------------|---------------|--------------|
| Naive persistence | 0.882 | 0.889 | 0.935 |
| ARIMA | 1.020 | 1.022 | 1.090 |
| TFT (pretrained) | 0.964 | 0.960 | 0.974 |
| **TFT + Residual (ours)** | **0.732** | **0.738** | **0.805** |

- **14.8% capacity-normalised NRMSE** — competitive with supervised methods requiring weather station data
- **< 1% difference** between pretrained and finetuned variants — zero-shot deployment viable
- **7-day recovery** after sudden 41.5% partial shading event
- Evaluated across **6 conditions**: clean × drifted × shaded, finetuned × pretrained

---

## Framework Overview

```
PVGIS Synthetic Data (10 European cities)
        │
        ▼
┌─────────────────────┐
│  TFT Foundation     │  ← pretrained, frozen at deployment
│  Model              │    never sees target household
└──────────┬──────────┘
           │  Q10 / Q50 / Q90
           ▼
┌─────────────────────────────────┐
│  Rolling MLP Residual           │  ← retrained daily on recent errors
│  Corrector Ensemble             │    30, 60, 90 day windows
│  (inverse-RMSE weighted)        │    adapts to drift & shading
└──────────────┬──────────────────┘
               │  Corrected Q10 / Q50 / Q90
               ▼
┌─────────────────────────────────┐
│  Stochastic MPC Battery         │  ← CVXPY / Gurobi
│  Controller                     │    scenario-based MILP
└─────────────────────────────────┘
```

**Stage 1 — TFT Foundation Model**
- Pretrained on synthetic PVGIS irradiance data from 10 European cities
- Produces calibrated Q10/Q50/Q90 quantile forecasts
- Never exposed to target household data at any stage

**Stage 2 — Rolling MLP Residual Corrector Ensemble**
- Lightweight MLP retrained daily on recent TFT forecast errors
- Temporally-bagged ensemble across 30, 60, 90 day rolling windows
- Inverse-RMSE weighted combination — longer windows upweighted in stable conditions
- Adapts to both gradual soiling drift and sudden partial shading

**Stage 3 — Stochastic MPC Battery Controller**
- Scenario-based MILP formulation using Q10/Q50/Q90 as scenarios
- Shared binary charge/discharge decisions with scenario-specific recourse variables
- Robust SOC back-off constraints under forecast uncertainty
- Optimises household energy cost against UK time-of-use tariffs

---

## Distribution Shift Robustness

ADAPT-PV is evaluated under two real-world panel degradation scenarios:

**Gradual Soiling Drift** — 25% linear power reduction from June 1st simulating dust/soiling accumulation. The corrector demonstrates *proactive adaptation*, maintaining below-baseline NRMSE from event onset as the rolling window incrementally tracks the decline.

**Sudden Partial Shading** — 41.5% step power reduction from June 1st (50% panel area shaded, empirically validated by Sarkar et al. 2024). The corrector demonstrates *reactive adaptation*, recovering below pre-event baseline within 7 days. TFT alone never recovers within the 30-day evaluation window.

---

## Installation

```bash
git clone https://github.com/AlyHafez/adaptive-pv-forecasting.git
cd adaptive-pv-forecasting

conda create -n pv-forecast python=3.11
conda activate pv-forecast
pip install -r requirements.txt
```



> **Note:** NumPy is pinned to `<2.0.0` due to PyTorch 2.2.x compatibility on Apple Silicon. On Linux x86 with PyTorch 2.4+ this constraint can be relaxed.

### Environment Variables

Create a `.env` file in the project root:

```bash
HF_TOKEN=your_huggingface_token   # required for UK_PV dataset
WANDB_API_KEY=your_wandb_key      # required for experiment tracking
```

---

## Project Structure

```
adaptive-pv-forecasting/
├── src/
│   ├── config.py                    # file paths, hyperparameters
│   ├── data/
│   │   ├── data_acquisition.py      # PVGIS API, UK_PV dataset, Open-Meteo
│   │   ├── data_processing.py       # cleaning, loading
│   │   └── feature_engineering.py  # temporal features, normalisation
│   ├── models/
│   │   ├── tft.py                   # TFT architecture
│   │   ├── train.py                 # pretraining on PVGIS
│   │   ├── transfer_learning.py     # cold-start finetuning
│   │   ├── eval.py                  # TFT evaluation, drift/shading simulation
│   │   ├── residual_corrector.py    # MLP architecture
│   │   ├── train_residuals.py       # rolling window retraining
│   │   └── eval_residual.py         # ensemble evaluation, recovery analysis
│   ├── evaluation/
│   │   ├── metrics.py               # MAE, RMSE, NRMSE, pinball, coverage
│   │   └── control.py               # stochastic MPC battery controller
│   └── utils/
│       └── utils.py                 # WandB login, seed setting
├── requirements.txt
|__ results/                            # where model checkpoints and csv predictions will load
|__data/
|   |–– raw/
|   |––processed/  # data will be loaded and split hwere into raw data straight from pvgis and processed which has features
|   
├── environment.yml
└── README.md
```

---

## Usage

### Step 1 — Data preparation
```bash
# download PVGIS synthetic data
python -m src.data.download_pvgis

# prepare UK_PV household data
python -m src.data.feature_engineering
```

### Step 2 — Pretraining on PVGIS
```bash
python -m src.models.train
```

### Step 3 — Evaluate TFT (zero-shot, no household data)
```bash
python -m src.models.eval ukpv_pretrained
```

### Step 4 — Optional finetuning
```bash
python -m src.models.transfer_learning
python -m src.models.eval ukpv_finetuned
```

### Step 5 — Residual corrector evaluation
```bash
# clean conditions
python -m src.models.eval_residual pretrained
python -m src.models.eval_residual finetuned

# gradual soiling drift
python -m src.models.eval ukpv_pretrained_drifted
python -m src.models.eval_residual pretrained_drifted

# sudden partial shading
python -m src.models.eval ukpv_pretrained_shaded
python -m src.models.eval_residual pretrained_shaded
```

### Step 6 — MPC economic evaluation
```bash
python -m src.evaluation.control
```

All runs log metrics, plots and recovery analysis to [Weights & Biases](https://wandb.ai).

---

## Probabilistic Metrics

| Metric | TFT | TFT+Residual | Better? |
|--------|-----|--------------|---------| 
| Pinball Q50 | 0.074 | 0.055 | ✓ +26% |
| Pinball Mean | 0.044 | 0.040 | ✓ +9% |
| Coverage (PICP) | 0.72 | 0.74 | △ improving |
| Interval Width | 0.473 | 0.466 | ✓ narrower |



---

## Ablation: Window Size Selection

| Configuration | NRMSE | vs Best Individual |
|---------------|-------|--------------------|
| 7-day only | 0.850 | worse |
| 30-day only | 0.780 | worse |
| 60-day only | 0.750 | — (best individual) |
| 5 windows (7–90) | 0.753 | worse |
| **3 windows (30+60+90)** | **0.740** | **better ✓** |

Short windows add noise not signal. The 3-window ensemble outperforms every individual window — the theoretically correct ensemble property.

---

## Citation

```bibtex
@mastersthesis{hafez2026adaptpv,
  title   = {ADAPT-PV: Adaptive Residential Solar PV Forecasting with Distribution Shift Robustness},
  author  = {Hafez, Aly},
  school  = {University College London},
  year    = {2026},
  note    = {Industry partnership with Fuse Energy}
}
```

---

## Acknowledgements

- [Fuse Energy](https://fuseenergy.com) for industry partnership and problem motivation
- [UK_PV dataset](https://www.ukpvlive.com) for real residential household data
- [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/) for synthetic pretraining data
- Supervisor: Chen Boli, UCL

---

## License

MIT License — see [LICENSE](LICENSE) for details.