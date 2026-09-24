# IT3051 Mini Project - Hotel Booking Cancellation Prediction

## Project Overview

This repository contains the IT3051 Fundamentals of Data Mining mini project on predicting hotel booking cancellations using machine learning.

The main objective is to predict whether a hotel reservation will be cancelled using booking-related information available before the final booking outcome is known.

This is a **binary classification** problem.

---

## Dataset

The project uses the **Hotel Booking Demand** dataset.

- Records: 119,390
- Columns: 32
- Target variable: `is_canceled`

Target classes:

- `0` = Not Cancelled
- `1` = Cancelled

The raw dataset is stored in:

```text
data/raw/hotel_bookings.csv
```

---

## Current Progress

The project has currently completed the initial **Exploratory Data Analysis (EDA)** stage.

Completed work includes:

- Dataset loading and inspection
- Target variable analysis
- Missing-value analysis
- Duplicate detection
- Numerical feature analysis
- Potential outlier investigation
- Logical data-quality checks
- Cancellation relationship analysis
- EDA visualizations
- EDA result tables

---

## Key EDA Findings

- 75,166 bookings were not cancelled, representing approximately 62.96% of the dataset.
- 44,224 bookings were cancelled, representing approximately 37.04%.
- The target variable does not show severe class imbalance.
- City Hotel bookings had a higher cancellation rate than Resort Hotel bookings.
- Cancelled bookings had higher lead times on average than non-cancelled bookings.
- Customers with previous cancellations showed a much higher cancellation rate.
- Repeated guests had a lower cancellation rate than non-repeated guests.
- Cancellation rates varied across market segments and customer types.
- Missing values were identified in `company`, `agent`, `country`, and `children`.
- 31,994 duplicate rows were detected and require further investigation.
- Several unusual numerical values were identified, including extreme ADR values and unusual guest counts.
- 180 bookings contained zero adults, zero children, and zero babies and will be reviewed during preprocessing.

These findings represent associations in the dataset and do not prove causation.

---

## Data Leakage Considerations

Some features may contain information that would not be available at the intended prediction time.

Important examples include:

```text
reservation_status
reservation_status_date
booking_changes
assigned_room_type
days_in_waiting_list
```

These features will be reviewed during preprocessing and feature selection to prevent data leakage.

---

## Repository Structure

```text
data/
├── raw/
└── processed/

notebooks/
└── eda/
    └── member1_eda.ipynb

src/
└── eda/
    └── check_dataset.py

results/
└── eda/
    ├── figures/
    └── tables/

README.md
requirements.txt
.gitignore
```

---

## EDA Outputs

Generated EDA figures are stored in:

```text
results/eda/figures/
```

Generated EDA summary tables are stored in:

```text
results/eda/tables/
```

The main EDA notebook is:

```text
notebooks/eda/member1_eda.ipynb
```

---

## Environment Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
pip install -r requirements.txt
```

Open the EDA notebook and select the project `.venv` as the Jupyter kernel.

---

## Current EDA Branch

```text
feature/it23824874-eda
```

This branch contains the initial dataset exploration, target analysis, EDA figures, tables, and supporting scripts.

---

## Next Steps

The next stages of the project include:

- Missing-value handling
- Duplicate handling
- Outlier treatment
- Data leakage prevention
- Feature engineering
- Feature selection
- Categorical encoding
- Train/test splitting
- Preprocessing pipeline development
- Model development and evaluation
