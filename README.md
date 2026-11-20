# SOH-prediction-using-NASA-Dataset

## Using python and scikit-learn

Feature analysis was done on the data and relevant features were identified. Further, we fit a Support Vector Regression model to predict the State of Health(SOH) 
and Remaining Useful Life(RUL) of the Li-ion battery with an accuracy of 70%. Batteries B0005, B0006, B0007 and B0018 were taken as test batteries.

Future scope: Applying a deep learning model to improve the model accuracy.

## Extended NASA battery dataset

The preprocessing scripts discover the original batteries and the uploaded
NASA battery folders automatically. The duplicate B0025-B0028 files are
deduplicated by battery ID.

1. Open `nasa_battery_life_prediction_dataset_cleaning.ipynb` in Jupyter and
   run its cells to convert each MATLAB cycle to a separate CSV under `tmp/`
   and append its metadata to `metadata.csv`. The notebook preserves existing
   metadata and skips battery IDs already present.
2. Run `python src/prep.py` to extract discharge-cycle features for SOH and RUL.
3. Run `python src/final.py` to train and evaluate the SOH and RUL models.

SOH uses valid discharge capacities normalized against the 2 Ah nominal
capacity. RUL uses the EOL capacity documented for each battery group, and
only batteries with a usable pre-EOL history and an observed endpoint are
included. Experiments with no documented EOL threshold, software-crash
endpoints, or suspect low-capacity measurements are excluded from supervised
RUL labels rather than assigned a guessed lifetime. Results and the final
models are written to `results/`.
