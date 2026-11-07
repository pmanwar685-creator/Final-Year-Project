# Final RUL model: observed-EOL cross-battery benchmark

## Model chosen

`ExtraTreesRegressor` was selected from three predeclared small-data candidates (Extra Trees, Random Forest, and Gradient Boosting) using lowest mean leave-one-battery-out (LOBO) MAE. It uses 37 measurements/features per cycle: cycle number, charge/discharge voltage/time/temperature/current features, plus causal 5-cycle trailing means. The five-cycle average at cycle *k* uses only cycles up to *k*.

This is intentionally a compact tree model rather than an LSTM/Transformer. There are only 331 valid supervised RUL cycles from three batteries; flexible deep networks are much more likely to overfit this amount of cell-level data. Recent reviews flag inconsistent EOL definitions, limited cells, and lack of cross-battery validation as major reasons why reported RUL accuracy is often not comparable ([review](https://www.sciencedirect.com/science/article/pii/S2095495626003426)). Gaussian-process and hybrid methods are promising for small battery datasets, but our Gaussian-process benchmark performed worse on this dataset; the selected ensemble is the best measured candidate here.

## Data-validity rules

- EOL means first measured capacity `<= 1.4 Ah` (70% of 2.0 Ah nominal capacity).
- B0005, B0006 and B0018 have observed EOL at cycles 125, 109 and 97 respectively.
- B0007 never reaches 1.4 Ah in the available record. It is right-censored and excluded from supervised RUL training and score calculation—not given a guessed failure cycle.
- The evaluation is battery-level LOBO: fit on two cells and predict the third unseen cell. A held-out battery is not used anywhere in that fold’s fit.

## Results

| Candidate | Mean MAE (cycles) | Mean RMSE (cycles) | Mean R² | Decision |
|---|---:|---:|---:|---|
| **Extra Trees** | **9.06** | 11.20 | 0.869 | Selected: best MAE |
| Random Forest | 9.26 | 11.21 | 0.863 | Close alternative |
| Gradient Boosting | 9.14 | **10.82** | **0.881** | Better RMSE/R², but higher MAE |

Selected Extra Trees fold scores:

| Held-out cell | MAE (cycles) | RMSE (cycles) | R² | Bias (cycles) |
|---|---:|---:|---:|---:|
| B0005 | 8.77 | 11.09 | 0.906 | -8.07 |
| B0006 | 8.61 | 10.25 | 0.894 | -0.67 |
| B0018 | 9.80 | 12.25 | 0.809 | +7.11 |

The maximum absolute error is 23.91 cycles averaged across folds. MAPE is reported in the JSON only for non-zero RUL because percentage error is mathematically undefined at EOL (RUL=0).

## Honest interpretation

This is a clean internal cross-battery benchmark, not a claim of “best in the literature.” Model selection and scoring use the same three cells, so more cells or nested LOBO are required for an independent final performance estimate. Many published NASA RUL numbers use a different and easier setting: train on early cycles of the same battery and predict its later cycles. Those numbers should not be ranked against this LOBO score. A recent physics-driven GPR study also uses LOBO and highlights uncertainty intervals as valuable for deployment ([paper](https://jemi.etmchina.com/jemien/article/abstract/B2508549)); uncertainty quantification is the recommended next upgrade after collecting additional cells.

## Run it

```bash
python src/prep.py
python src/train_rul_final.py
```

Outputs:

- `results/rul_observed_eol_results.json` — fold-by-fold metrics and model comparison
- `results/models/rul_observed_eol_model.joblib` — final Extra Trees model, feature list, data scope and EOL threshold
