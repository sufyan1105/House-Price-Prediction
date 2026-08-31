# 🖥️ Live Prediction Web App

An interactive single-page front-end for the trained Random Forest model.

```bash
pip install -r requirements.txt
python app.py            # first run trains + caches model.pkl / pipeline.pkl (~30s)
```

Then open **http://127.0.0.1:5000**

---

## What it does

| | |
|---|---|
| **Interactive map** | Click or drag anywhere on California to set `longitude` / `latitude`. 3,000 sampled training blocks are plotted underneath, coloured by their real `median_house_value`. |
| **Live estimate** | Every input change re-runs the model and animates to the new number. |
| **Uncertainty band** | Each of the 100 trees votes separately; the 10th–90th percentile of those votes becomes the low/high range. |
| **Reality check** | The actual recorded median value of the 30 nearest real blocks in `housing.csv`, next to the estimate — an instant sanity check. |
| **Contribution bars** | For each feature, how far the estimate moves when that feature is reset to its dataset median. Longitude and latitude are grouped as one "where it is" factor, since neither means anything alone. |
| **Smart auto-fill** | `total_rooms`, `total_bedrooms`, `population`, `households` and `housing_median_age` describe an entire census block, not one house. With auto-fill on, they are pulled from the 30 nearest real blocks whenever the pin moves. Toggle it off to drive them by hand. |
| **Type any value** | Every readout — including latitude and longitude — is click-to-edit. Sliders cover the 5th–95th percentile for usable resolution; typing accepts anything within the dataset's full observed range, and the readout turns amber when a typed value sits outside the slider's window. Enter commits, Escape cancels. |
| **Per-home mode** | See below. |
| **$500k ceiling warning** | The training labels are clipped at $500,001, so the model cannot predict above it. The UI says so when an estimate approaches the cap. |

---

## Per-home mode — and what this dataset does *not* contain

Nobody knows their block's `total_rooms`. The **Per home** tab under *Block-level detail*
lets you work in per-household terms instead:

| You set | Derived as |
|---|---|
| Rooms per home | `total_rooms = rooms_per_home × households` |
| Bedrooms per home | `total_bedrooms = bedrooms_per_home × households` |
| People per home | `population = people_per_home × households` |
| Homes in the block | scales all three totals, holding the ratios fixed |

A strip under the sliders always shows the raw block totals the model will actually receive.

**Two honest caveats, both surfaced in the UI:**

1. **There is no square footage in this dataset.** Not in any column. "Rooms per home"
   (dataset mean ≈ 5.4) is the closest available proxy for house size.
2. **"Bedrooms per home" is not a listing's bedroom count.** It is
   `total_bedrooms / households` from census aggregates, and it sits near **1.05**
   across California — the dataset counts these differently from how an estate agent would.
   Treat it as a relative dial, not "a 3-bed house".

These ratios are *not* model features — the model still eats the same eight raw columns.
They are a UI convenience that converts back to block totals before the request is sent,
so `main.py` and `app.py` remain compatible with the identical `model.pkl`.

For reference, the ratios rank like this against `median_house_value`:
`rooms_per_household` **+0.15**, `bedrooms_per_room` **−0.26** — versus `median_income` at **+0.69**,
which is why income dominates the contribution bars.

---

## Files

| File | Role |
|---|---|
| `app.py` | Flask server. Training logic is identical to `main.py`, so both share the same `model.pkl` / `pipeline.pkl`. |
| `templates/index.html` | The entire front-end — HTML, CSS and JS in one file, no CDN, works offline. |
| `requirements.txt` | Dependencies. |

Nothing in the original project was modified. `main.py` still trains and runs batch
inference on `input.csv` exactly as before.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | Feature ranges, per-household ratio ranges, categories, scatter points, model info |
| `GET /api/autofill?longitude=&latitude=` | Median block stats of the 30 nearest real blocks |
| `POST /api/predict` | `{...9 features}` → `{price, low, high, std, percentile, capped, contributions}` |

---

## Note on `.gitignore`

`model.pkl` is already ignored. `pipeline.pkl` is **not** — consider adding it,
since both are large binaries that regenerate on first run.
