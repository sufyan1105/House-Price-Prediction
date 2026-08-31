# 🏠 House Price Prediction

A machine learning project that predicts median house values from the California Housing dataset — taken all the way from raw data to a **live, interactive web app**.

The project covers the full ML pipeline: exploratory analysis, preprocessing, model selection, a production `scikit-learn` pipeline, and a Flask front-end where you drop a pin on a map and watch the model re-estimate in real time.

---

## ✨ Live Estimator

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000**

> First launch trains the model and caches `model.pkl` / `pipeline.pkl` (~30 s).
> Every launch after that is instant.

| Feature | What it does |
|---|---|
| 🗺️ **Interactive map** | Click or drag anywhere on California to set `longitude` / `latitude`. 3,000 sampled training blocks sit underneath, coloured by their real median value. |
| ⚡ **Live estimate** | Every input change re-runs the model and animates to the new number. |
| 📊 **Uncertainty band** | All 100 trees vote individually — the 10th–90th percentile of their votes becomes the low/high range. |
| ✅ **Reality check** | The *actual* recorded median of the 30 nearest real blocks, shown beside the estimate. An instant accuracy read. |
| 🔍 **Contribution bars** | How far the estimate moves when each feature is reset to its dataset median. `longitude` + `latitude` are grouped, since neither means anything alone. |
| 🏘️ **Smart auto-fill** | Block-level features are pulled from the 30 nearest real blocks whenever the pin moves. Toggle off to drive them yourself. |
| ⌨️ **Type any value** | Every readout is click-to-edit, latitude and longitude included. |
| 🏡 **Per-home mode** | Work in per-household terms (rooms per home, people per home) instead of raw block totals. |

Full front-end documentation lives in **[`WEBAPP.md`](WEBAPP.md)**.

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── app.py                             # 🌐 Flask server for the live estimator
├── templates/
│   └── index.html                     # Single-page UI (HTML + CSS + JS, no CDN)
├── requirements.txt                   # Dependencies
├── WEBAPP.md                          # Web app documentation
│
├── main.py                            # Train model + batch inference on input.csv
├── model_testing.py                   # Model comparison via cross-validation
│
├── Analyzing the data.ipynb           # Exploratory data analysis
├── visualizing the data.ipynb         # Geographic + correlation plots
├── Handling categorical attributes.ipynb  # One-hot encoding ocean_proximity
├── Feature Scaling.ipynb              # Imputation → encoding → standardisation
├── Further Preprocessing.ipynb        # Imputer deep-dive
├── Sklearn Pipeline.ipynb             # Composing the numeric Pipeline
├── TestingData.ipynb                  # Stratified split verification
│
├── housing.csv                        # California Housing dataset (20,640 rows)
├── input.csv                          # Sample input for batch inference
├── input - Comparing.csv              # Input data for model comparison
└── output.csv                         # Batch predictions
```

---

## 🔍 How It Works

### The pipeline

Both `main.py` and `app.py` build the **same** `ColumnTransformer`, so they share one set of model artefacts:

```
numeric (8 cols) ──► SimpleImputer(median) ──► StandardScaler
                                                              ├──► RandomForestRegressor
ocean_proximity  ──► OneHotEncoder(handle_unknown="ignore") ──┘
```

Data is split 80/20 with `StratifiedShuffleSplit` on a binned `income_cat`, so the
income distribution of the training set matches the population — a plain random split
would skew it.

### Two ways to run it

| | Entry point | Behaviour |
|---|---|---|
| **Web** | `app.py` | Serves the UI, predicts one block at a time on demand |
| **Batch** | `main.py` | Reads `input.csv` → writes `output.csv` |

Both auto-train on first run if `model.pkl` is missing, then load from cache.

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `longitude` / `latitude` | Geographic coordinates |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` | Total rooms across the block |
| `total_bedrooms` | Total bedrooms (**207 missing values** — median-imputed) |
| `population` | Block population |
| `households` | Number of households |
| `median_income` | Median household income, in units of $10,000 |
| `ocean_proximity` | Categorical: `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND` |
| `median_house_value` | **Target variable** |

### Two things worth knowing

**1. The target is capped at $500,001.** 992 rows — 4.8% of the dataset — sit on that ceiling,
so the model cannot predict above it. Estimates near the cap are floors, not truths —
the web app flags this when it happens.

**2. Every row describes a census block, not a house.** `total_rooms` and `total_bedrooms`
are block aggregates, and there is no square footage anywhere in the data. The prediction
is a block's *median* value.

### Correlations with `median_house_value`

| Feature | Correlation |
|---|---|
| `median_income` | **+0.69** |
| `rooms_per_household` (derived) | +0.15 |
| `total_rooms` | +0.13 |
| `housing_median_age` | +0.11 |
| `latitude` | −0.14 |
| `bedrooms_per_room` (derived) | **−0.26** |

Income dominates by a wide margin — which is exactly what the contribution bars in the app show.

---

## 🧪 Model Selection

`model_testing.py` compares three regressors using 10-fold cross-validated RMSE:

| Model | Train RMSE | CV RMSE | Verdict |
|---|---:|---:|---|
| `LinearRegression` | $69,051 | $69,204 ± 2,372 | Underfits — train and CV are identical, so more data won't help |
| `DecisionTreeRegressor` | **$0** | $69,081 ± 2,296 | Memorises the training set perfectly, generalises no better than a straight line |
| `RandomForestRegressor` | $18,342 | **$49,432 ± 2,125** | **Selected** — ~29% lower error than either baseline |

```bash
python model_testing.py
```

The decision tree is the instructive one: a training RMSE of exactly **$0** looks like a
flawless model until cross-validation puts it level with linear regression. That gap
between train and CV error is the entire argument for cross-validating.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/sufyan1105/House-Price-Prediction.git
cd House-Price-Prediction
pip install -r requirements.txt
```

### Run the web app

```bash
python app.py          # → http://127.0.0.1:5000
```

### Run batch inference

```bash
python main.py         # input.csv → output.csv
```

### Explore the notebooks

```bash
jupyter notebook
```

Recommended order:

1. `Analyzing the data.ipynb`
2. `visualizing the data.ipynb`
3. `Handling categorical attributes.ipynb`
4. `Feature Scaling.ipynb`
5. `Further Preprocessing.ipynb`
6. `Sklearn Pipeline.ipynb`
7. `TestingData.ipynb`

---

## 🧠 Model Summary

| Detail | Value |
|---|---|
| Algorithm | `RandomForestRegressor(random_state=42)` |
| Trees | 100 |
| Training rows | 16,512 (80% stratified split) |
| Evaluation metric | RMSE |
| Cross-validated RMSE | **~$49,400** (10-fold) |
| Validation strategy | 10-fold cross-validation |
| Preprocessing | Median imputation → standard scaling → one-hot encoding |

---

## 📦 Dependencies

```
flask · pandas · numpy · scikit-learn · joblib
```

Plus `jupyter` for the notebooks. All pinned in `requirements.txt`.

> `model.pkl` and `pipeline.pkl` are gitignored — they regenerate on first run.

---

## 👤 Author

**Sufyan** — [github.com/sufyan1105](https://github.com/sufyan1105)
