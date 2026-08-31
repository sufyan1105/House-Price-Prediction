"""
House Price Prediction — Web App
================================
A Flask server that wraps the trained RandomForest pipeline from this project
and serves an interactive single-page UI for live price estimation.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000

Notes
-----
* Training logic is intentionally identical to `main.py` so the web app and the
  batch script share the exact same model artefacts (model.pkl / pipeline.pkl).
* If model.pkl / pipeline.pkl are missing, they are trained on first launch
  (takes ~20-40s) and cached to disk.
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "housing.csv")
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
PIPELINE_FILE = os.path.join(BASE_DIR, "pipeline.pkl")

NUM_ATTRIBS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
CAT_ATTRIBS = ["ocean_proximity"]
FEATURE_ORDER = NUM_ATTRIBS + CAT_ATTRIBS


# --------------------------------------------------------------------------- #
#  Training  (mirrors main.py)
# --------------------------------------------------------------------------- #
def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])


def stratified_train_split(housing):
    """Same 80/20 stratified-by-income split used across the notebooks."""
    housing = housing.copy()
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(split.split(housing, housing["income_cat"]))
    return housing.loc[train_idx].drop("income_cat", axis=1)


def load_or_train():
    raw = pd.read_csv(DATA_FILE)

    if os.path.exists(MODEL_FILE) and os.path.exists(PIPELINE_FILE):
        model = joblib.load(MODEL_FILE)
        pipeline = joblib.load(PIPELINE_FILE)
        print("[model] loaded model.pkl + pipeline.pkl")
    else:
        print("[model] no saved model found — training RandomForestRegressor ...")
        train = stratified_train_split(raw)
        labels = train["median_house_value"].copy()
        features = train.drop("median_house_value", axis=1)

        pipeline = build_pipeline(NUM_ATTRIBS, CAT_ATTRIBS)
        prepared = pipeline.fit_transform(features)

        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        model.fit(prepared, labels)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(pipeline, PIPELINE_FILE)
        print("[model] trained and cached to model.pkl / pipeline.pkl")

    return raw, model, pipeline


RAW, MODEL, PIPELINE = load_or_train()

# Reference statistics used by the UI (slider bounds, defaults, autofill).
STATS = {
    col: {
        "min": float(RAW[col].min()),
        "max": float(RAW[col].max()),
        "median": float(RAW[col].median()),
        "p05": float(RAW[col].quantile(0.05)),
        "p95": float(RAW[col].quantile(0.95)),
    }
    for col in NUM_ATTRIBS
}

# Per-household ratios. These are NOT model features — the model still eats the
# eight raw columns — but they are a far more human way to describe a block, so
# the UI lets you drive them and back-derives the totals from `households`.
_ratios = pd.DataFrame({
    "rooms_per_home": RAW.total_rooms / RAW.households,
    "bedrooms_per_home": RAW.total_bedrooms / RAW.households,
    "people_per_home": RAW.population / RAW.households,
})
RATIO_STATS = {
    col: {
        "min": float(_ratios[col].quantile(0.005)),
        "max": float(_ratios[col].quantile(0.995)),
        "median": float(_ratios[col].median()),
    }
    for col in _ratios.columns
}

OCEAN_CATEGORIES = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
MODE_OCEAN = RAW["ocean_proximity"].mode()[0]

# Downsampled scatter for the map (lon, lat, value) — keeps the payload small.
_scatter = RAW.sample(n=min(3000, len(RAW)), random_state=7)
SCATTER = [
    [round(float(r.longitude), 3), round(float(r.latitude), 3), int(r.median_house_value)]
    for r in _scatter.itertuples()
]

# KD-ish lookup arrays for the neighbourhood autofill.
_COORDS = RAW[["longitude", "latitude"]].to_numpy()
_BLOCK = RAW[["total_rooms", "total_bedrooms", "population", "households",
              "housing_median_age", "median_income", "median_house_value"]].to_numpy()


app = Flask(__name__)


# --------------------------------------------------------------------------- #
#  Prediction helpers
# --------------------------------------------------------------------------- #
def to_frame(payload):
    row = {}
    for col in NUM_ATTRIBS:
        row[col] = float(payload.get(col, STATS[col]["median"]))
    prox = payload.get("ocean_proximity", MODE_OCEAN)
    row["ocean_proximity"] = prox if prox in OCEAN_CATEGORIES else MODE_OCEAN
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def predict_rows(df):
    return MODEL.predict(PIPELINE.transform(df))


def tree_spread(df):
    """Per-tree predictions -> a genuine uncertainty band for this input."""
    X = PIPELINE.transform(df)
    preds = np.array([est.predict(X)[0] for est in MODEL.estimators_])
    return float(preds.mean()), float(preds.std()), preds


def contributions(df):
    """
    Local sensitivity: for each feature, how much does the estimate move when
    that single feature is reset to its dataset-median (or modal) value?
    Positive => this input pushes the price up vs. a typical California block.
    """
    base = float(predict_rows(df)[0])
    variants, names = [], []

    # Longitude + latitude only mean anything together, so they move as one group.
    groups = [("location", ["longitude", "latitude"])] + [
        (c, [c]) for c in NUM_ATTRIBS if c not in ("longitude", "latitude")
    ]

    for name, cols in groups:
        alt = df.copy()
        for c in cols:
            alt.loc[:, c] = STATS[c]["median"]
        variants.append(alt)
        names.append(name)

    alt = df.copy()
    alt.loc[:, "ocean_proximity"] = MODE_OCEAN
    variants.append(alt)
    names.append("ocean_proximity")

    batch = pd.concat(variants, ignore_index=True)
    neutral = predict_rows(batch)

    out = [{"feature": n, "delta": float(base - v)} for n, v in zip(names, neutral)]
    out.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return out


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def meta():
    return jsonify({
        "stats": STATS,
        "ratios": RATIO_STATS,
        "ocean_categories": OCEAN_CATEGORIES,
        "scatter": SCATTER,
        "target": {
            "min": float(RAW["median_house_value"].min()),
            "max": float(RAW["median_house_value"].max()),
            "median": float(RAW["median_house_value"].median()),
        },
        "n_rows": int(len(RAW)),
        "n_trees": len(MODEL.estimators_),
    })


@app.route("/api/autofill")
def autofill():
    """Median block statistics of the 30 nearest census blocks to a point."""
    lon = float(request.args.get("longitude", -119.5))
    lat = float(request.args.get("latitude", 35.6))

    d2 = (_COORDS[:, 0] - lon) ** 2 + (_COORDS[:, 1] - lat) ** 2
    idx = np.argpartition(d2, 30)[:30]
    med = np.nanmedian(_BLOCK[idx], axis=0)
    # A tiny cluster can be all-NaN for total_bedrooms; fall back to the global median.
    globals_ = np.array([STATS[c]["median"] for c in
                         ["total_rooms", "total_bedrooms", "population", "households",
                          "housing_median_age", "median_income"]] + [200000.0])
    med = np.where(np.isnan(med), globals_, med)

    nearest = int(np.argmin(d2))
    return jsonify({
        "total_rooms": float(med[0]),
        "total_bedrooms": float(med[1]),
        "population": float(med[2]),
        "households": float(med[3]),
        "housing_median_age": float(med[4]),
        "median_income": float(med[5]),
        "neighbourhood_median_value": float(med[6]),
        "ocean_proximity": str(RAW.iloc[nearest]["ocean_proximity"]),
        "distance_deg": float(np.sqrt(d2[nearest])),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}
    df = to_frame(payload)

    mean, std, preds = tree_spread(df)
    lo, hi = np.percentile(preds, [10, 90])

    resp = {
        "price": mean,
        "std": std,
        "low": float(lo),
        "high": float(hi),
        "capped": bool(mean >= 480000),
    }

    if payload.get("explain", True):
        resp["contributions"] = contributions(df)

    # Where does this sit in the state-wide distribution?
    resp["percentile"] = float(
        (RAW["median_house_value"] < mean).mean() * 100
    )
    return jsonify(resp)


if __name__ == "__main__":
    print("\n  ➜  House Price Prediction running at  http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
