# Hotel Booking Cancellation Prediction

University Data Mining mini-project for predicting the binary target
`is_canceled` (`0` = not cancelled, `1` = cancelled).

## Progress Evaluation 1 scope

The repository currently covers exploratory data analysis, data-quality handling,
leakage investigation, feature engineering, and a reusable preprocessing pipeline.
It intentionally stops before model training, tuning, evaluation, and application
development.

## Data and project flow

The source dataset is `data/raw/hotel_bookings.csv` (119,390 rows and 32 columns).
Run the notebooks in this order:

1. `notebooks/eda/member1_eda.ipynb`
2. `notebooks/data_quality/member2_data_quality.ipynb`
3. `notebooks/feature_engineering/member3_feature_engineering.ipynb`
4. `notebooks/preprocessing/member4_preprocessing.ipynb`

Each notebook uses repository-relative paths. Member 4 recreates the approved
Member 2 and Member 3 handoff through the reusable functions under `src/`, so it
does not depend on hidden notebook state.

## Reproduce the Evaluation 1 workflow

Install the pinned environment from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Execute each notebook from a clean kernel:

```powershell
python -m nbconvert --to notebook --execute --inplace notebooks\eda\member1_eda.ipynb --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute --inplace notebooks\data_quality\member2_data_quality.ipynb --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute --inplace notebooks\feature_engineering\member3_feature_engineering.ipynb --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute --inplace notebooks\preprocessing\member4_preprocessing.ipynb --ExecutePreprocessor.timeout=600
```

Generated EDA, data-quality, and feature-validation outputs are organized under
`results/`. The final Member 4 notebook leaves `preprocessor`, the train/test
partitions, processed matrices, and processed feature names available for the
later modeling stage. It also generates the single compressed CSV handoff:

```text
data/processed/hotel_bookings_preprocessed.csv
```

The file contains `source_index`, `data_split`, 899 transformed feature columns,
and the unchanged `is_canceled` target. It can be loaded directly with:

```python
processed_df = pd.read_csv("data/processed/hotel_bookings_preprocessed.csv")
```
