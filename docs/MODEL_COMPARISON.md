# SoH and RUL model comparison

**Status: evidence for presentation, generated 8 November 2026.**  This document distinguishes a like-for-like result from an indicative one.  It must not be used to claim that an easier within-cell split beats an unseen-cell test.

## 1. What was evaluated

| Item | Our final pipeline | Earlier repository model | Prashik's new RUL work |
|---|---|---|---|
| Purpose | Joint SoH and RUL estimation | README says SVR for SoH/RUL | Builds a leakage-aware RUL training set; it is **not yet a trained/evaluated RUL model** |
| SoH estimator | Standardised Ridge regression | SVR (hyperparameters/held-out metrics not recorded) | — |
| RUL estimator | Gradient boosting over causal predicted-SoH trajectory features (`gap`, 20/40-cycle slopes, all-history slope) | Not reproducibly specified | — |
| Cycle-level inputs | Charge/discharge durations, voltage-level times, voltage, current and temperature summaries; 5-cycle trailing means | Not fully documented | Voltage drop/slope, temperature rise, and past impedance `Re`/`Rct` |
| Data | NASA PCoE B0005, B0006, B0007, B0018 | Same claimed cells | Same raw cells; labels only B0005, B0006, B0018 |
| Generalisation protocol | Leave-one-battery-out (LOBO): train on 3 cells, test on the unseen fourth cell | “70% accuracy” is not enough to identify a protocol | Explicitly warns to use battery-level splits |

The source data are the NASA PCoE 18650-cell runs: 2.0 Ah nominal capacity, 1.5 A CC/CV charge and 2 A discharge, with cut-off voltage varying by cell.  The paper linked below gives a concise description of these exact B0005/B0006/B0007/B0018 runs and the common 70% end-of-life convention: [NASA-data description](https://pmc.ncbi.nlm.nih.gov/articles/PMC12464235/).

## 2. Our reproducible results

### SoH: valid headline result

SoH is `100 × measured capacity / 2.0 Ah`.  The result below is the mean of four LOBO folds.  A test battery is never used to fit either scaler or Ridge model.

| Metric | Mean LOBO result | Per-cell range |
|---|---:|---:|
| MAE | **0.495 percentage points** | 0.239–0.956 pp |
| RMSE | **0.550 percentage points** | 0.283–0.990 pp |
| R² | **0.9950** | 0.9848–0.9987 |

Per-cell SoH MAE/RMSE/R²: B0005 0.279/0.348/0.9987; B0006 0.506/0.579/0.9979; B0007 0.956/0.990/0.9848; B0018 0.239/0.283/0.9987.  Exact values and plots are in `results/results.json` and `results/plots/lobo_results.png`.

### RUL: current status—do not oversell it

The older final-script result is MAE **13.78 cycles**, RMSE **18.07 cycles**, R² **0.756** under LOBO.  It is an *exploratory* result only, not the presentation headline, because that script assigns B0007's last recorded cycle as EOL.  In reality B0007 never reaches 1.4 Ah in the observed data, so its EOL is right-censored.

Prashik's dataset preparation fixes this label issue correctly:

| Cell | EOL at 1.4 Ah | RUL-labelled cycles | Status |
|---|---:|---:|---|
| B0005 | 125 | 125 | observed EOL |
| B0006 | 109 | 109 | observed EOL |
| B0018 | 97 | 97 | observed EOL |
| B0007 | — | 0 | right-censored; excluded |

It produces 331 valid labelled rows and uses the most recent impedance measurement at or before each discharge, avoiding future-measurement leakage.  The next defensible RUL headline is therefore: retrain and evaluate with **only B0005/B0006/B0018**, using three-fold LOBO, training-only imputation for `Re`/`Rct`, and report fold-level MAE/RMSE/R² plus error-at-prediction-start.  Do not quote the 13.78-cycle value as a fair comparison to literature until this rerun is complete.

## 3. Literature comparison

### Directly comparable SoH study

The closest published comparison found uses the same four NASA cells and reports a leave-one-out cross-validation (LOO-CV) experiment.  Its thresholds and feature details still differ, so this is a strong but not perfect comparison.

| Method | Validation | Mean MAE (pp) | Mean RMSE (pp) | Mean R² | Comparison with our Ridge LOBO |
|---|---|---:|---:|---:|---|
| **Our Ridge + causal cycle features** | 4-cell LOBO | **0.495** | **0.550** | **0.9950** | — |
| Different-charging-rate model (2025) | 4-cell LOO-CV | 1.758 | 2.309 | 0.9880 | Our MAE is **71.8% lower**; RMSE **76.2% lower** |

The published LOO-CV cell metrics used for that mean are MAE 1.4832/2.8076/1.2854/1.4574%, RMSE 2.0271/3.7736/1.6013/1.8330%, and R² 0.9899/0.9917/0.9826/0.9878 for B0005/B0006/B0007/B0018 respectively ([paper](https://doi.org/10.1016/j.fub.2025.100074)).  Percentage improvements are calculated from unrounded means: `(literature − ours) / literature`.

### Why several impressive papers are not directly comparable

| Paper/method | Reported result | Why it must not be ranked directly against ours |
|---|---|---|
| Transformer-embedded SOC/SoH model | NASA cell-wise SoH MAE 0.0016–0.0024 (fractional SoH), R² 0.9834–0.9937 | Different model task/joint SOC setting and reported split; it is not a four-cell LOBO aggregate. [Paper](https://onlinelibrary.wiley.com/doi/full/10.1007/s12598-024-02942-z) |
| CNN–BiLSTM–attention SoH | NASA B0005/B0006/B0018, reports RMSE/MAE/MAPE by prediction start | Prediction-start protocol and selected cells differ. [Paper](https://www.mdpi.com/2313-0105/8/10/155) |
| MSFFCNN RUL | Claims small RUL/capacity errors on the same NASA family | Uses within-cell historical data (e.g. 70% train/30% test) and changes B0007's failure threshold; this leaks cell identity/degradation trajectory relative to LOBO. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12464235/) |
| Dual-attention hybrid RUL | Reports B0005/B0006/B0007/B0018 RUL MAE 1.589/3.323/1.765/3.453 | Uses specified prediction-start points rather than held-out unseen cells; excellent interpolation is not cross-cell generalisation. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12796292/) |

## 4. Presentation-ready claims

Use these claims verbatim, with the caveats intact:

1. **“On a strict unseen-battery test, our lightweight Ridge SoH model achieved 0.495 percentage-point MAE, 0.550 percentage-point RMSE and R²=0.995 across all four NASA cells.”**
2. **“Against the closest four-cell LOO-CV study we found, our mean SoH MAE is 71.8% lower and our RMSE is 76.2% lower; this is a like-for-like cross-cell comparison, although feature extraction differs.”**
3. **“The model is intentionally lightweight: it outperformed our small-data LSTM experiment (1.75 pp SoH MAE) while requiring no GPU and remaining inspectable for BMS deployment.”**
4. **“For RUL, we identified and corrected a data-validity issue: B0007 is right-censored at 1.4 Ah. We will report the final RUL score only after observed-EOL, battery-level LOBO validation.”**

Never say “best in literature,” “70% accuracy,” or compare raw RUL MAE to a paper unless EOL threshold, unit, prediction horizon and split protocol all match.

## 5. Metrics to show in the final RUL rerun

For every LOBO fold report: MAE (cycles), RMSE (cycles), median absolute error, max absolute error, R², signed bias `mean(pred−true)`, under-prediction and over-prediction rates, and relative error at fixed prediction starts (for example 30, 50 and 70 cycles).  Also show: number of train/test cells and cycles, EOL threshold, features available at inference, missing-value policy, model size/training time, and a true-vs-predicted trajectory plot per held-out cell.  MAPE is unstable at EOL because true RUL is zero; either omit it or state an explicit non-zero-cycle exclusion rule.

## 6. Reproducibility

```bash
python src/prep.py
python src/final.py
```

The current SoH and exploratory RUL outputs are written to `results/results.json`; model and feature definitions are in `src/final.py`, `src/train.py`, and `src/rul_exp2.py`.  The clean RUL-label construction is in `rul_dataset.ipynb`, with outputs `rul_prediction_dataset.csv` and `rul_eol_summary.csv`.
