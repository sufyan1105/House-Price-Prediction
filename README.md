# 🏠 House Price Prediction

A machine learning project that predicts median house values using the California Housing dataset. The project walks through the full ML pipeline — from data analysis and preprocessing to model training and inference — using scikit-learn and Jupyter Notebooks.

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── Analyzing the data.ipynb           # Exploratory data analysis
├── visualizing the data.ipynb         # Data visualization
├── Handling categorical attributes.ipynb  # Encoding categorical features
├── Feature Scaling.ipynb              # Normalization and scaling
├── Further Preprocessing.ipynb        # Advanced preprocessing steps
├── Sklearn Pipeline.ipynb             # Building the full sklearn pipeline
├── TestingData.ipynb                  # Model evaluation on test data
│
├── model_testing.py                   # Model evaluation script
├── main.py                            # Train model and run inference
│
├── housing.csv                        # California Housing dataset
├── input.csv                          # Sample input for inference
├── input - Comparing.csv              # Input data for model comparison
└── output.csv                         # Predicted house values
```

---

## 🔍 How It Works

The project is split into two phases:

**Training Phase** (runs automatically if no saved model is found):
1. Loads `housing.csv` and creates an `income_cat` column for stratified splitting.
2. Performs an 80/20 stratified train/test split.
3. Builds a `ColumnTransformer` pipeline:
   - Numerical features → `SimpleImputer` (median) + `StandardScaler`
   - Categorical features → `OneHotEncoder`
4. Trains a `RandomForestRegressor` on the prepared data.
5. Saves the model to `model.pkl` and the pipeline to `pipeline.pkl`.

**Inference Phase** (runs if `model.pkl` already exists):
1. Loads the saved model and pipeline.
2. Reads `input.csv`, transforms it through the pipeline.
3. Predicts `median_house_value` and writes results to `output.csv`.

---

## 📊 Dataset

The project uses the **California Housing dataset** (`housing.csv`), which includes:

| Feature | Description |
|---|---|
| `longitude` / `latitude` | Geographic coordinates |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Block population |
| `households` | Number of households |
| `median_income` | Median income of households |
| `ocean_proximity` | Categorical: distance to the ocean |
| `median_house_value` | **Target variable** |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/sufyan1105/House-Price-Prediction.git
cd House-Price-Prediction
pip install numpy pandas scikit-learn joblib
```

### Run Training & Inference

```bash
python main.py
```

- First run: trains the model and saves `model.pkl` and `pipeline.pkl`.
- Subsequent runs: loads the saved model and generates predictions to `output.csv`.

### Run in Jupyter

Open any notebook in order to explore the step-by-step analysis:

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

## 🧪 Model

| Detail | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Evaluation Metric | RMSE (Root Mean Squared Error) |
| Validation Strategy | Cross-validation |
| Train/Test Split | 80% / 20% (stratified by income category) |

---

## 📦 Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `jupyter` (for notebooks)

---

## 👤 Author

**Sufyan** — [github.com/sufyan1105](https://github.com/sufyan1105)
