"""
Per-Country Error Analysis — SVR RBF, Quota Deviation as DV
============================================================
Runs the same rolling time-series cross-validation as SVRDeviation2.py,
but after each fold records R², RMSE, and MAE broken down by country.

Outputs:
    country_fold_errors.csv   — long-format: fold × country × metric
    fig_country_errors.png    — facet grid: R², RMSE, MAE over folds per country

Place this script in the same folder as SVRDeviation2.py and your raw data files.
Run with:  python SVRDeviation_CountryErrorAnalysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# ── 0. Configuration (must match SVRDeviation2.py) ───────────────────────────

PROD_CSV   = "COil_production_monthly.csv"
BRENT_XLSX = "DCOILBRENTEU.xlsx"
GPR_XLS    = "GPR.xls"
VIX_XLSX   = "VIXCLS.xlsx"
OILGDP_XLS = "Oil_%_of_GDP.xls"
INST_XLSX  = "Institutional_Quality_Index.xlsx"
QUOTA_XLSX = "OPEC_Quotas.xlsx"

OPEC = {
    "SAU": "Saudi Arabia", "IRN": "Iran",    "IRQ": "Iraq",
    "KWT": "Kuwait",       "ARE": "UAE",      "VEN": "Venezuela",
    "NGA": "Nigeria",      "DZA": "Algeria",  "LBY": "Libya",
    "GAB": "Gabon",        "COG": "Congo",    "GNQ": "Eq. Guinea",
}
OPEC_NAME_MAP = {
    "Algeria": "DZA", "Congo": "COG", "Equatorial Guinea": "GNQ",
    "Gabon": "GAB", "IR Iran": "IRN", "Iraq": "IRQ", "Kuwait": "KWT",
    "Libya": "LBY", "Nigeria": "NGA", "Saudi Arabia": "SAU",
    "United Arab Emirates": "ARE", "Venezuela": "VEN",
}
REF_COUNTRY = "SAU"

WINDOW_ROWS    = 616
STEP_ROWS      = 103
PARAM_GRID     = {
    "C":       [1, 10, 50, 100, 200, 500],
    "epsilon": [0.01, 0.1, 0.5, 1.0, 2.0],
    "gamma":   ["scale", "auto", 0.01, 0.001],
}
N_INNER_SPLITS = 3
N_JOBS         = -1
MIN_OBS        = 5   # minimum test obs per country to compute metrics


# ── 1. Data loading (identical to SVRDeviation2.py) ──────────────────────────

def load_production():
    raw = pd.read_csv(PROD_CSV, sep=";", header=None, low_memory=False)
    records = []
    for pair in range(raw.shape[1] // 2):
        dc, vc = pair * 2, pair * 2 + 1
        key  = str(raw.iloc[1, vc])
        iso3 = next((c for c in OPEC if f"-{c}-" in key), None)
        if iso3 is None or "INTL.53-1-" not in key:
            continue
        records.append(pd.DataFrame({
            "date": raw.iloc[9:, dc].astype(str),
            "prod": pd.to_numeric(raw.iloc[9:, vc], errors="coerce"),
            "iso3": iso3,
        }))
    df = pd.concat(records).dropna()
    df = df[df["date"].str.match(r"\d{4}-\d{2}")]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    return df

def load_quota():
    ql = pd.read_excel(QUOTA_XLSX, sheet_name="Long Format (for analysis)")
    ql["Period_Start"] = pd.to_datetime(ql["Period_Start"].astype(str) + "-01")
    ql["Period_End"]   = pd.to_datetime(ql["Period_End"].astype(str)   + "-01")
    rows = []
    for _, row in ql.iterrows():
        if not isinstance(row["Allocation_kbd"], (int, float, np.integer)):
            continue
        val = float(row["Allocation_kbd"])
        if np.isnan(val):
            continue
        iso3 = OPEC_NAME_MAP.get(row["Country"])
        if iso3 is None:
            continue
        for m in pd.date_range(row["Period_Start"], row["Period_End"], freq="MS"):
            rows.append({"date": m, "iso3": iso3, "quota": val})
    return pd.DataFrame(rows)

def monthly_mean(path, col, engine=None):
    kw = {"engine": engine} if engine else {}
    df = pd.read_excel(path, parse_dates=[0], **kw)
    df.columns = ["date", col]
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.groupby("date")[col].mean().reset_index()

def load_gpr():
    df = pd.read_excel(GPR_XLS, engine="xlrd")[["month", "GPR"]]
    df.columns = ["date", "gpr"]
    df["date"] = (pd.to_datetime(df["date"], errors="coerce")
                  .dt.to_period("M").dt.to_timestamp())
    return df.dropna()

def load_oil_gdp():
    df = pd.read_excel(OILGDP_XLS, engine="xlrd", header=3)
    df = df.rename(columns={"Country Code": "iso3"})
    yc = [c for c in df.columns if str(c).isdigit() and 1960 <= int(c) <= 2030]
    df = df[["iso3"] + yc].melt(id_vars="iso3", var_name="year", value_name="oil_gdp")
    df["year"] = df["year"].astype(int)
    return df[df["iso3"].isin(OPEC)]

def load_inst():
    df = pd.read_excel(INST_XLSX)
    nm = {
        "Algeria": "DZA", "Equatorial Guinea": "GNQ", "Gabon": "GAB",
        "Iran": "IRN", "Iraq": "IRQ", "Kuwait": "KWT", "Libya": "LBY",
        "Nigeria": "NGA", "Saudi Arabia": "SAU", "United Arab Emirates": "ARE",
        "Venezuela": "VEN", "Congo": "COG",
    }
    df["iso3"] = df["Country"].map(nm)
    df = df.dropna(subset=["iso3"])
    wgi = [c for c in df.columns if any(k in c for k in
           ["effectiveness", "corruption", "Regulatory", "Political stability"])]
    df["inst"] = df[wgi].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df[["iso3", "year", "inst"]]

def build_shocks(brent, vix, gpr):
    df = (brent.merge(vix, on="date", how="inner")
               .sort_values("date").reset_index(drop=True))
    df["d_ln_brent"] = np.log(df["brent"]).diff()
    df = df.dropna()
    arma = ARIMA(df["vix"].values, order=(1, 0, 1)).fit()
    df["ORS"] = arma.resid
    X = np.column_stack([np.ones(len(df)), df["ORS"].values])
    coef, *_ = np.linalg.lstsq(X, df["d_ln_brent"].values, rcond=None)
    df["ODS"] = df["d_ln_brent"].values - X @ coef
    assert abs(df["ORS"].corr(df["ODS"])) < 1e-6, "ORS and ODS not orthogonal"
    for col in ["ORS", "ODS"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    g = gpr.sort_values("date").reset_index(drop=True)
    g["ln_gpr"] = np.log(g["gpr"])
    g["lag"]    = g["ln_gpr"].shift(1)
    g = g.dropna()
    c = np.polyfit(g["lag"], g["ln_gpr"], 1)
    resid = g["ln_gpr"] - (c[0] * g["lag"] + c[1])
    g["GPR_shock"] = (resid - resid.mean()) / resid.std()
    return (df[["date", "ORS", "ODS"]]
            .merge(g[["date", "GPR_shock"]], on="date", how="inner"))


# ── 2. Build panel ────────────────────────────────────────────────────────────

print("Loading data …")
prod    = load_production()
quota_m = load_quota()
brent   = monthly_mean(BRENT_XLSX, "brent")
vix     = monthly_mean(VIX_XLSX,   "vix")
gpr_raw = load_gpr()
oil_gdp = load_oil_gdp()
inst    = load_inst()

print("Constructing shocks …")
shocks = build_shocks(brent, vix, gpr_raw)
shocks = shocks.merge(vix, on="date", how="left")

merged = prod.merge(quota_m, on=["iso3", "date"], how="inner")
merged = merged[merged["quota"] >= merged["prod"] * 0.4]
merged["dev_pct"] = (merged["prod"] - merged["quota"]) / merged["quota"] * 100

panel = merged.merge(shocks, on="date", how="inner")
panel["year"] = panel["date"].dt.year
panel = (panel.merge(oil_gdp, on=["iso3", "year"], how="left")
              .merge(inst,    on=["iso3", "year"], how="left"))
panel = panel.sort_values(["iso3", "date"]).reset_index(drop=True)
for col in ["oil_gdp", "inst"]:
    panel[col] = panel.groupby("iso3")[col].ffill().bfill()
panel["ln_vix"] = np.log(panel["vix"])

for shock in ["ORS", "ODS", "GPR_shock"]:
    panel[f"{shock}_12m"] = panel.groupby("iso3")[shock].transform(
        lambda x: x.rolling(12, min_periods=6).sum())
panel["ODS_l6"]     = panel.groupby("iso3")["ODS_12m"].shift(6)
panel["ORS_l6"]     = panel.groupby("iso3")["ORS_12m"].shift(6)
panel["GPR_l12"]    = panel.groupby("iso3")["GPR_shock_12m"].shift(12)
panel["ln_vix_12m"] = panel.groupby("iso3")["ln_vix"].transform(
    lambda x: x.rolling(12, min_periods=6).mean())
panel["dev_pct_l1"] = panel.groupby("iso3")["dev_pct"].shift(1)

for iso3 in OPEC:
    if iso3 != REF_COUNTRY:
        panel[f"C_{iso3}"] = (panel["iso3"] == iso3).astype(float)

SHOCK_F = ["ODS_l6", "ORS_l6", "GPR_l12"]
CTRL_F  = ["ln_vix_12m", "oil_gdp", "inst", "dev_pct_l1"]
DUMMY_F = [f"C_{c}" for c in OPEC if c != REF_COUNTRY]
ALL_F   = SHOCK_F + CTRL_F + DUMMY_F
N_CONT  = len(SHOCK_F) + len(CTRL_F)

KEY = ["dev_pct"] + SHOCK_F + CTRL_F
p = panel.dropna(subset=KEY).copy()
p = p.sort_values(["date", "iso3"]).reset_index(drop=True)

X_raw   = p[ALL_F].values.astype(float)
y       = p["dev_pct"].values.astype(float)
dates   = p["date"].values
iso3s   = p["iso3"].values   # ← track country for each row

print(f"Panel: {len(p):,} obs | {p['iso3'].nunique()} countries | "
      f"{p['date'].min().date()} – {p['date'].max().date()}")


# ── 3. Rolling CV — per-country metrics ──────────────────────────────────────

n           = len(p)
fold_starts = [s for s in range(0, n - WINDOW_ROWS - STEP_ROWS + 1, STEP_ROWS)
               if s + WINDOW_ROWS < n]

print(f"\nTotal folds: {len(fold_starts)}")
print(f"Min obs per country to report metrics: {MIN_OBS}\n")
print(f"{'Fold':>4}  {'Test period':>13}  {'Country':>7}  "
      f"{'N_te':>5}  {'RMSE':>8}  {'MAE':>8}")
print("─" * 70)

country_fold_records = []

for fold, start in enumerate(fold_starts, start=1):
    end_train  = start + WINDOW_ROWS
    start_test = end_train
    end_test   = min(start_test + STEP_ROWS, n)

    tr_idx = np.arange(start, end_train)
    te_idx = np.arange(start_test, end_test)

    X_tr = X_raw[tr_idx].copy()
    X_te = X_raw[te_idx].copy()
    y_tr = y[tr_idx]
    y_te = y[te_idx]

    sc = StandardScaler()
    X_tr[:, :N_CONT] = sc.fit_transform(X_tr[:, :N_CONT])
    X_te[:, :N_CONT] = sc.transform(X_te[:, :N_CONT])

    inner_cv = TimeSeriesSplit(n_splits=N_INNER_SPLITS)
    gs = GridSearchCV(
        SVR(kernel="rbf"), PARAM_GRID,
        cv=inner_cv,
        n_jobs=N_JOBS, refit=True, verbose=0,
    )
    gs.fit(X_tr, y_tr)

    y_pred = gs.best_estimator_.predict(X_te)

    d_te_from = pd.Timestamp(dates[te_idx[0]]).strftime("%Y-%m")
    d_te_to   = pd.Timestamp(dates[te_idx[-1]]).strftime("%Y-%m")
    test_period = f"{d_te_from}–{d_te_to}"

    # ── Per-country breakdown within this fold's test window ──
    te_countries = iso3s[te_idx]
    for country in np.unique(te_countries):
        mask = te_countries == country
        n_c  = mask.sum()
        if n_c < MIN_OBS:
            continue
        y_c    = y_te[mask]
        yp_c   = y_pred[mask]
        # R² undefined / misleading if all true values are constant
        rmse_c = np.sqrt(mean_squared_error(y_c, yp_c))
        mae_c  = mean_absolute_error(y_c, yp_c)

        country_fold_records.append({
            "fold":        fold,
            "test_from":   d_te_from,
            "test_to":     d_te_to,
            "iso3":        country,
            "country":     OPEC.get(country, country),
            "n_test":      int(n_c),
            "rmse":        round(rmse_c, 4),
            "mae":         round(mae_c, 4),
            "best_C":      gs.best_params_["C"],
            "best_eps":    gs.best_params_["epsilon"],
            "best_gamma":  gs.best_params_["gamma"],
        })
        print(f"  {fold:>4}  {test_period:>13}  {country:>7}  "
              f"{n_c:>5}  {rmse_c:>8.2f}  {mae_c:>8.2f}")

results_df = pd.DataFrame(country_fold_records)
results_df.to_csv("country_fold_errors.csv", index=False)
print(f"\nSaved: country_fold_errors.csv  ({len(results_df)} rows)")


# ── 4. Summary table: mean metrics per country across folds ──────────────────

summary = (results_df
           .groupby(["iso3", "country"])[["rmse", "mae"]]
           .agg(["mean", "std"])
           .round(4))
summary.columns = ["rmse_mean", "rmse_std", "mae_mean", "mae_std"]
summary = summary.reset_index().sort_values("rmse_mean")

print(f"\n{'─'*75}")
print("Country summary (averaged across folds where N_test ≥ MIN_OBS):")
print(f"{'─'*75}")
print(f"  {'ISO':>5}  {'Country':<16}"
      f"{'RMSE mean':>10}  {'MAE mean':>9}")
print(f"  {'─'*70}")
for _, row in summary.iterrows():
    print(f"{row['iso3']:>5}  {row['country']:<16} {row['rmse_mean']:>10.2f}  {row['mae_mean']:>9.2f}")


# ── 5. Figure: per-country RMSE, MAE over folds ──────────────────────────

countries  = sorted(results_df["iso3"].unique())
n_countries = len(countries)
metrics    = [("tab:blue"),
              ("rmse", "RMSE (pp)", "tab:orange"),
              ("mae",  "MAE (pp)",  "tab:green")]

fig, axes = plt.subplots(
    nrows=n_countries, ncols=3,
    figsize=(15, 2.4 * n_countries),
    sharex=False, sharey=False,
)
# Ensure 2-D indexing even if n_countries == 1
if n_countries == 1:
    axes = axes[np.newaxis, :]

for row_i, iso3 in enumerate(countries):
    cdf = results_df[results_df["iso3"] == iso3].sort_values("fold")
    cname = OPEC.get(iso3, iso3)

    for col_j, (metric, label, color) in enumerate(metrics):
        ax = axes[row_i, col_j]
        vals = cdf[metric].values
        folds = cdf["fold"].values

        ax.plot(folds, vals, marker="o", color=color,
                linewidth=1.6, markersize=4, alpha=0.85)
        ax.axhline(np.nanmean(vals), color=color, linewidth=0.9,
                   linestyle="--", alpha=0.5)


        # Row label on the left-most column only
        if col_j == 0:
            ax.set_ylabel(cname, fontsize=9, rotation=0, labelpad=55,
                          va="center", ha="right")

        # Column header on the top row only
        if row_i == 0:
            ax.set_title(label, fontsize=10, fontweight="bold")

        ax.set_xlabel("Fold", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(folds)

        # Annotate mean
        ax.text(0.97, 0.05, f"μ={np.nanmean(vals):.2f}",
                transform=ax.transAxes, fontsize=6.5,
                ha="right", va="bottom", color=color, alpha=0.8)

fig.suptitle(
    "Per-Country SVR Error Analysis — Rolling Time-Series CV\n"
    "(dashed line = mean across folds)",
    fontsize=11, fontweight="bold", y=1.005,
)
plt.tight_layout()
plt.savefig("fig_country_errors.png", dpi=160, bbox_inches="tight")
print("Saved: fig_country_errors.png")