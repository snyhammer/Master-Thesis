"""
OPEC Quota Deviation — Polynomial Regression Baseline (Libya excluded)
======================================================================
Baseline model comparable to SVR. Identical data pipeline, shock
construction, panel structure, country set and rolling CV setup as
SVRDeviation_noLBY.py. Libya is excluded throughout.

Model:
    dev_pct = b0
            + b1*ODS_l6  + b2*ODS_l6^2
            + b3*ORS_l6  + b4*ORS_l6^2
            + b5*GPR_l12 + b6*GPR_l12^2
            + b7*ln_vix_12m + b8*oil_gdp + b9*inst + b10*dev_pct_l1
            + country dummies (SAU = reference)
            + e

Validation:
    Same rolling TSCV as SVR: window=616 rows, step=103 rows, 11 folds.
    Scaler fitted on training window only — no look-ahead.

Outputs:
    results_poly_dev.csv           — fold metrics + coefficients
    fig_poly_dev_metrics.png       — R², RMSE, MAE bar chart
    fig_poly_dev_coefficients.png  — shock coefficient plot

Requirements:
    pip install pandas numpy scikit-learn statsmodels matplotlib openpyxl xlrd
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# ── 0. Configuration ───────────────────────────────────────────────────────────
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
PROD_CSV   = os.path.join(DATA_DIR, "COil_production_monthly.csv")
BRENT_XLSX = os.path.join(DATA_DIR, "DCOILBRENTEU.xlsx")
GPR_XLS    = os.path.join(DATA_DIR, "GPR.xls")
VIX_XLSX   = os.path.join(DATA_DIR, "VIXCLS.xlsx")
OILGDP_XLS = os.path.join(DATA_DIR, "Oil_%_of_GDP.xls")
INST_XLSX  = os.path.join(DATA_DIR, "Institutional_Quality_Index.xlsx")
QUOTA_XLSX = os.path.join(DATA_DIR, "OPEC_Quotas.xlsx")

# Libya excluded
OPEC = {
    "SAU":"Saudi Arabia","IRN":"Iran","IRQ":"Iraq","KWT":"Kuwait",
    "ARE":"UAE","VEN":"Venezuela","NGA":"Nigeria","DZA":"Algeria",
    "GAB":"Gabon","COG":"Congo","GNQ":"Eq. Guinea",
}
OPEC_NAME_MAP = {
    "Algeria":"DZA","Congo":"COG","Equatorial Guinea":"GNQ","Gabon":"GAB",
    "IR Iran":"IRN","Iraq":"IRQ","Kuwait":"KWT",
    "Nigeria":"NGA","Saudi Arabia":"SAU","United Arab Emirates":"ARE","Venezuela":"VEN",
}
REF_COUNTRY = "SAU"

# Rolling CV — identical to SVR script
WINDOW_ROWS = 616
STEP_ROWS   = 103

SHOCK_F = ["ODS_l6", "ORS_l6", "GPR_l12"]
CTRL_F  = ["ln_vix_12m", "oil_gdp", "inst", "dev_pct_l1"]

# ── 1. Data loaders (identical to SVRDeviation_noLBY.py) ──────────────────────

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
    df  = pd.read_excel(INST_XLSX)
    nm  = {
        "Algeria":"DZA","Equatorial Guinea":"GNQ","Gabon":"GAB","Iran":"IRN",
        "Iraq":"IRQ","Kuwait":"KWT","Nigeria":"NGA",
        "Saudi Arabia":"SAU","United Arab Emirates":"ARE","Venezuela":"VEN","Congo":"COG",
    }
    df["iso3"] = df["Country"].map(nm)
    df = df.dropna(subset=["iso3"])
    wgi = [c for c in df.columns if any(k in c for k in
           ["effectiveness","corruption","Regulatory","Political stability"])]
    df["inst"] = df[wgi].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df[["iso3","year","inst"]]

# ── 2. Shock construction (Ready, 2018) — identical to SVR script ─────────────

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
    for col in ["ORS","ODS"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    g = gpr.sort_values("date").reset_index(drop=True)
    g["ln_gpr"] = np.log(g["gpr"])
    g["lag"]    = g["ln_gpr"].shift(1)
    g = g.dropna()
    c = np.polyfit(g["lag"], g["ln_gpr"], 1)
    resid = g["ln_gpr"] - (c[0] * g["lag"] + c[1])
    g["GPR_shock"] = (resid - resid.mean()) / resid.std()
    return df[["date","ORS","ODS"]].merge(g[["date","GPR_shock"]], on="date", how="inner")

# ── 3. Build panel ─────────────────────────────────────────────────────────────

print("Loading data ...")
prod    = load_production()
quota_m = load_quota()
brent   = monthly_mean(BRENT_XLSX, "brent")
vix     = monthly_mean(VIX_XLSX,   "vix")
gpr_raw = load_gpr()
oil_gdp = load_oil_gdp()
inst    = load_inst()

print("Constructing shocks (Ready, 2018) ...")
shocks = build_shocks(brent, vix, gpr_raw)
shocks = shocks.merge(vix, on="date", how="left")
print("  ORS ⊥ ODS verified  ✓")

merged = prod.merge(quota_m, on=["iso3","date"], how="inner")
merged = merged[merged["quota"] >= merged["prod"] * 0.4]
merged["dev_pct"] = (merged["prod"] - merged["quota"]) / merged["quota"] * 100

panel = merged.merge(shocks, on="date", how="inner")
panel["year"] = panel["date"].dt.year
panel = (panel.merge(oil_gdp, on=["iso3","year"], how="left")
              .merge(inst,    on=["iso3","year"], how="left"))
panel = panel.sort_values(["iso3","date"]).reset_index(drop=True)
for col in ["oil_gdp","inst"]:
    panel[col] = panel.groupby("iso3")[col].ffill().bfill()
panel["ln_vix"] = np.log(panel["vix"])

for shock in ["ORS","ODS","GPR_shock"]:
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

DUMMY_F = [f"C_{c}" for c in OPEC if c != REF_COUNTRY]
ALL_F   = SHOCK_F + CTRL_F + DUMMY_F
N_CONT  = len(SHOCK_F) + len(CTRL_F)

KEY = ["dev_pct"] + ALL_F
p = panel.dropna(subset=KEY).copy()
p = p.sort_values(["date","iso3"]).reset_index(drop=True)

print(f"\nPanel: {len(p):,} obs | {p['iso3'].nunique()} countries | "
      f"{p['date'].min().date()} – {p['date'].max().date()}")
print(f"DV: mean={p['dev_pct'].mean():+.2f}%  std={p['dev_pct'].std():.2f}%  "
      f"range [{p['dev_pct'].min():.1f}%, {p['dev_pct'].max():.1f}%]")

# ── 4. Polynomial feature matrix ───────────────────────────────────────────────
# Add squared shock terms alongside linear terms.
# All continuous features (including squares) scaled per training window.

X_raw  = p[ALL_F].values.astype(float)
y      = p["dev_pct"].values.astype(float)
n      = len(p)
dates  = p["date"].values

shock_idx    = [ALL_F.index(s) for s in SHOCK_F]
X_sq         = X_raw[:, shock_idx] ** 2
X_poly       = np.hstack([X_raw, X_sq])
feat_names   = ALL_F + [f"{s}^2" for s in SHOCK_F]
N_CONT_POLY  = N_CONT + len(SHOCK_F)   # continuous cols to scale

# ── 5. Rolling Time Series Cross-Validation ────────────────────────────────────

fold_starts = range(0, n - WINDOW_ROWS - STEP_ROWS + 1, STEP_ROWS)
n_folds     = sum(1 for s in fold_starts if s + WINDOW_ROWS < n)

print(f"\nRolling CV: window={WINDOW_ROWS} rows, step={STEP_ROWS} rows, "
      f"{n_folds} folds")
print(f"\n{'Fold':>4}  {'Test period':>20}  {'R²':>8}  {'RMSE':>8}  {'MAE':>8}")
print("─" * 56)

fold_results = []
all_oos_true = []
all_oos_pred = []

for fold, start in enumerate(fold_starts, start=1):
    end_train  = start + WINDOW_ROWS
    start_test = end_train
    end_test   = min(start_test + STEP_ROWS, n)
    if start_test >= n:
        break

    tr_idx = np.arange(start, end_train)
    te_idx = np.arange(start_test, end_test)

    X_tr = X_poly[tr_idx].copy()
    X_te = X_poly[te_idx].copy()
    y_tr = y[tr_idx]
    y_te = y[te_idx]

    sc = StandardScaler()
    X_tr[:, :N_CONT_POLY] = sc.fit_transform(X_tr[:, :N_CONT_POLY])
    X_te[:, :N_CONT_POLY] = sc.transform(X_te[:, :N_CONT_POLY])

    mdl    = LinearRegression().fit(X_tr, y_tr)
    y_pred = mdl.predict(X_te)

    r2   = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)

    d_te_from = pd.Timestamp(dates[te_idx[0]]).strftime("%Y-%m")
    d_te_to   = pd.Timestamp(dates[te_idx[-1]]).strftime("%Y-%m")

    fold_results.append({
        "fold": fold, "test_from": d_te_from, "test_to": d_te_to,
        "n_train": len(tr_idx), "n_test": len(te_idx),
        "r2_test": r2, "rmse_test": rmse, "mae_test": mae,
    })
    all_oos_true.extend(y_te)
    all_oos_pred.extend(y_pred)

    print(f"{fold:>4}  {d_te_from}–{d_te_to}  {r2:>+8.4f}  "
          f"{rmse:>8.2f}pp  {mae:>8.2f}pp")

fold_df  = pd.DataFrame(fold_results)
cv_r2    = fold_df["r2_test"].mean()
cv_rmse  = fold_df["rmse_test"].mean()
cv_mae   = fold_df["mae_test"].mean()
oos_r2   = r2_score(all_oos_true, all_oos_pred)
oos_rmse = np.sqrt(mean_squared_error(all_oos_true, all_oos_pred))
oos_mae  = mean_absolute_error(all_oos_true, all_oos_pred)

# ── 6. Final model on full data ────────────────────────────────────────────────

sc_full = StandardScaler()
X_full  = X_poly.copy()
X_full[:, :N_CONT_POLY] = sc_full.fit_transform(X_poly[:, :N_CONT_POLY])
final   = LinearRegression().fit(X_full, y)
yis     = final.predict(X_full)
is_r2   = r2_score(y, yis)
is_rmse = np.sqrt(mean_squared_error(y, yis))
is_mae  = mean_absolute_error(y, yis)

# ── 7. Print summary ───────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("POLYNOMIAL REGRESSION — OPEC Deviation DV (Libya excluded)")
print(f"{'='*60}")
print(f"N: {len(p):,} obs | {p['iso3'].nunique()} countries | "
      f"{p['date'].min().date()} – {p['date'].max().date()}")
print(f"\n  {'Metric':<8}  {'In-sample':>12}  {'CV mean':>12}  {'OOS pooled':>12}")
print(f"  {'─'*50}")
print(f"  {'R²':<8}  {is_r2:>+12.4f}  {cv_r2:>+12.4f}  {oos_r2:>+12.4f}")
print(f"  {'RMSE':<8}  {is_rmse:>12.2f}  {cv_rmse:>12.2f}  {oos_rmse:>12.2f}  pp")
print(f"  {'MAE':<8}  {is_mae:>12.2f}  {cv_mae:>12.2f}  {oos_mae:>12.2f}  pp")

print(f"\n  Shock coefficients (standardised):")
for name, coef in zip(feat_names, final.coef_):
    if name in SHOCK_F + [f"{s}^2" for s in SHOCK_F]:
        print(f"    {name:<24s}: {coef:>+8.4f} pp")

# ── 8. Figure 1: Metrics bar chart ────────────────────────────────────────────

metrics   = ["R²", "RMSE (pp)", "MAE (pp)"]
is_vals   = [is_r2,  is_rmse,  is_mae]
oos_vals  = [oos_r2, oos_rmse, oos_mae]
colours   = {"In-sample": "#003153", "OOS pooled": "#c0392b"}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, metric, is_v, oos_v in zip(axes, metrics, is_vals, oos_vals):
    bars = ax.bar(
        list(colours.keys()), [is_v, oos_v],
        color=list(colours.values()), width=0.5, alpha=0.88,
    )
    for bar, val in zip(bars, [is_v, oos_v]):
        sign = "+" if metric == "R²" and val >= 0 else ""
        fmt  = f"{sign}{val:.3f}" if metric == "R²" else f"{val:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + abs(bar.get_height()) * 0.02,
            fmt, ha="center", va="bottom", fontsize=11,
            fontweight="medium", color="#1a1a1a",
        )
    ax.set_title(metric, fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    if metric != "R²":
        ax.set_ylim(bottom=0)

plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(DATA_DIR, "fig_poly_dev_metrics.png"),
            dpi=180, bbox_inches="tight")
print("\nSaved: fig_poly_dev_metrics.png")

# ── 9. Figure 2: Shock coefficients ───────────────────────────────────────────

shock_terms = SHOCK_F + [f"{s}^2" for s in SHOCK_F]
label_map = {
    "ODS_l6":    "ODS  (linear)",
    "ODS_l6^2":  "ODS  (squared)",
    "ORS_l6":    "ORS  (linear)",
    "ORS_l6^2":  "ORS  (squared)",
    "GPR_l12":   "GPR  (linear)",
    "GPR_l12^2": "GPR  (squared)",
}
colour_map = {
    "ODS_l6":    "#003153","ODS_l6^2":  "#6699bb",
    "ORS_l6":    "#c0392b","ORS_l6^2":  "#e07b72",
    "GPR_l12":   "#1a7a4a","GPR_l12^2": "#70bb8a",
}
order = ["ODS_l6","ODS_l6^2","ORS_l6","ORS_l6^2","GPR_l12","GPR_l12^2"]
coef_dict = dict(zip(feat_names, final.coef_))

fig2, ax2 = plt.subplots(figsize=(9, 5))
y_pos = np.arange(len(order))
vals  = [coef_dict[t] for t in order]
cols  = [colour_map[t] for t in order]
labs  = [label_map[t]  for t in order]

bars2 = ax2.barh(y_pos, vals, color=cols, height=0.55, alpha=0.88)
for bar, val in zip(bars2, vals):
    xpos = val + (0.005 if val >= 0 else -0.005)
    ha   = "left" if val >= 0 else "right"
    ax2.text(xpos, bar.get_y() + bar.get_height()/2,
             f"{val:+.4f}", va="center", ha=ha, fontsize=9.5, color="#1a1a1a")

ax2.axvline(0, color="black", linewidth=0.9)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labs, fontsize=11)
ax2.set_xlabel("Standardised coefficient — effect on quota deviation (pp)",
               fontsize=10, labelpad=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="x", alpha=0.2, linewidth=0.6)

from matplotlib.patches import Patch
ax2.legend(handles=[
    Patch(facecolor="#003153", alpha=0.88, label="Oil Demand Shock (ODS)"),
    Patch(facecolor="#c0392b", alpha=0.88, label="Oil Risk Shock (ORS)"),
    Patch(facecolor="#1a7a4a", alpha=0.88, label="Geopolitical Risk (GPR)"),
], loc="lower right", fontsize=9, frameon=True, framealpha=0.9)

plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(DATA_DIR, "fig_poly_dev_coefficients.png"),
            dpi=180, bbox_inches="tight")
print("Saved: fig_poly_dev_coefficients.png")

# ── 10. Export CSV ─────────────────────────────────────────────────────────────

rows = []
for _, r in fold_df.iterrows():
    rows.append({"type":"cv_fold","fold":r["fold"],
                 "test_from":r["test_from"],"test_to":r["test_to"],
                 "n_train":r["n_train"],"n_test":r["n_test"],
                 "r2":r["r2_test"],"rmse_pp":r["rmse_test"],"mae_pp":r["mae_test"]})
for label,rv,rmv,rmae in [
    ("in_sample",  is_r2,  is_rmse,  is_mae),
    ("cv_mean",    cv_r2,  cv_rmse,  cv_mae),
    ("oos_pooled", oos_r2, oos_rmse, oos_mae),
]:
    rows.append({"type":label,"r2":rv,"rmse_pp":rmv,"mae_pp":rmae})
for name,coef in zip(feat_names, final.coef_):
    rows.append({"type":"coefficient","feature":name,"coef_std":coef})

pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "results_poly_dev.csv"), index=False)
print("Saved: results_poly_dev.csv")

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")