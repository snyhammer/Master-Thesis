### Uses Deviation as DV ###


"""
OPEC Quota Deviation — SVR with RBF Kernel
==========================================
Dependent variable:
    dev_pct = (actual_production − quota) / quota × 100
    Positive = overproduction above quota.
    Not standardised — kept in percentage-point units.

Shock construction (Ready, 2018):
    ORS: ARMA(1,1) residuals of monthly VIX level
         → unexpected financial risk shock
    ODS: residuals of regressing Δln(Brent) on ORS
         → non-risk oil price shock (ORS ⊥ ODS by construction)
    GPR: AR(1) residuals of ln(GPR Index)
         → unexpected geopolitical risk

    All three standardised monthly to N(0,1), then summed into
    12-month rolling windows, then lagged:
        ODS: lag 6 months  (data-driven optimal from cross-correlogram)
        ORS: lag 6 months
        GPR: lag 12 months

Features (standardised to mean=0, std=1 on each training fold):
    ODS_l6, ORS_l6, GPR_l12, ln_vix_12m, oil_gdp, inst, dev_pct_l1
    + country dummies (SAU = reference, omitted)

Validation — Rolling Time Series Cross-Validation:
    Fixed training window (~6 years / 616 rows) rolls forward by
    ~1 year (103 rows) per fold. 13 folds total.
    Scaler and hyperparameters fitted on training window only.

Hyperparameter tuning:
    Nested inner 3-fold rolling CV within each training window.
    Grid: C ∈ {1, 10, 50, 100}, epsilon ∈ {0.1, 0.5, 1.0},
          gamma ∈ {scale, auto}

Outputs:
    results_svr_dev.csv          — fold metrics + importance
    fig_svr_dev.png              — actual vs predicted (clean, no legend)
    fig_svr_dev_importance.png   — permutation feature importance

Requirements:
    pip install pandas numpy scikit-learn matplotlib openpyxl xlrd
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from statsmodels.tsa.arima.model import ARIMA

# ── 0. Configuration ──────────────────────────────────────────────────────────
PROD_CSV   = "COil_production_monthly.csv"
BRENT_XLSX = "DCOILBRENTEU.xlsx"
GPR_XLS    = "GPR.xls"
VIX_XLSX   = "VIXCLS.xlsx"
OILGDP_XLS = "Oil_%_of_GDP.xls"
INST_XLSX  = "Institutional_Quality_Index.xlsx"
QUOTA_XLSX = "OPEC_Quotas.xlsx"

OPEC = {
    "SAU":"Saudi Arabia", "IRN":"Iran",    "IRQ":"Iraq",
    "KWT":"Kuwait",       "ARE":"UAE",      "VEN":"Venezuela",
    "NGA":"Nigeria",      "DZA":"Algeria", "LBY":"Libya",
    "GAB":"Gabon",        "COG":"Congo",   "GNQ":"Eq. Guinea",
}
OPEC_NAME_MAP = {
    "Algeria":"DZA","Congo":"COG","Equatorial Guinea":"GNQ","Gabon":"GAB",
    "IR Iran":"IRN","Iraq":"IRQ","Kuwait":"KWT","Libya":"LBY",
    "Nigeria":"NGA","Saudi Arabia":"SAU","United Arab Emirates":"ARE",
    "Venezuela":"VEN",
}
REF_COUNTRY = "SAU"

# Rolling CV parameters — chosen based on data structure:
#   Total obs = 1,960 | ~229 unique months | 2006–2016 gap in quota data
#   Window = 616 rows ≈ 6 years of data (enough to span 2–3 OPEC agreement cycles)
#   Step   = 103 rows ≈ 1 year (one OPEC agreement cycle)
#   → 13 folds, covering 2000–2023

WINDOW_ROWS = 616
STEP_ROWS   = 103

# Hyperparameter grid
PARAM_GRID = {
    "C":       [1, 10, 50, 100, 200, 500],
    "epsilon": [0.01,0.1, 0.5, 1.0, 2.0],
    "gamma":   ["scale", "auto", 0.01, 0.001],
}
N_INNER_SPLITS = 3   # inner rolling CV folds for hyperparameter tuning
N_PERM_REPS    = 30  # permutation importance repetitions
N_JOBS         = -1


# ── 1. Data loading ───────────────────────────────────────────────────────────

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
    df = df[["iso3"] + yc].melt(id_vars="iso3", var_name="year",
                                 value_name="oil_gdp")
    df["year"] = df["year"].astype(int)
    return df[df["iso3"].isin(OPEC)]

def load_inst():
    df  = pd.read_excel(INST_XLSX)
    nm  = {
        "Algeria":"DZA","Equatorial Guinea":"GNQ","Gabon":"GAB","Iran":"IRN",
        "Iraq":"IRQ","Kuwait":"KWT","Libya":"LBY","Nigeria":"NGA",
        "Saudi Arabia":"SAU","United Arab Emirates":"ARE","Venezuela":"VEN",
        "Congo":"COG",
    }
    df["iso3"] = df["Country"].map(nm)
    df = df.dropna(subset=["iso3"])
    wgi = [c for c in df.columns if any(k in c for k in
           ["effectiveness","corruption","Regulatory","Political stability"])]
    df["inst"] = df[wgi].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df[["iso3","year","inst"]]

# ── 2. Shock construction (Ready, 2018) ───────────────────────────────────────

def build_shocks(brent, vix, gpr):
    """
    ORS = ARMA(1,1) residuals of VIX — unexpected financial risk.
    ODS = Δln(Brent) residuals after regressing on ORS — non-risk price shock.
    GPR = AR(1) residuals of ln(GPR) — unexpected geopolitical shock.

    ORS ⊥ ODS by construction (correlation = 0).
    All standardised to N(0,1) at monthly frequency.
    """
    df = (brent.merge(vix, on="date", how="inner")
               .sort_values("date").reset_index(drop=True))
    df["d_ln_brent"] = np.log(df["brent"]).diff()
    df = df.dropna()

    # Step 1: ORS — unexpected VIX shock
    arma = ARIMA(df["vix"].values, order=(1, 0, 1)).fit()
    df["ORS"] = arma.resid

    # Step 2: ODS — Brent change orthogonalised to ORS
    X = np.column_stack([np.ones(len(df)), df["ORS"].values])
    coef, *_ = np.linalg.lstsq(X, df["d_ln_brent"].values, rcond=None)
    df["ODS"] = df["d_ln_brent"].values - X @ coef

    # Verify orthogonality
    assert abs(df["ORS"].corr(df["ODS"])) < 1e-6, "ORS and ODS not orthogonal"

    # Standardise monthly shocks to N(0,1)
    for col in ["ORS", "ODS"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Step 3: GPR shock — AR(1) residual of ln(GPR)
    g = gpr.sort_values("date").reset_index(drop=True)
    g["ln_gpr"] = np.log(g["gpr"])
    g["lag"]    = g["ln_gpr"].shift(1)
    g = g.dropna()
    c = np.polyfit(g["lag"], g["ln_gpr"], 1)
    resid = g["ln_gpr"] - (c[0] * g["lag"] + c[1])
    g["GPR_shock"] = (resid - resid.mean()) / resid.std()

    return (df[["date","ORS","ODS"]]
            .merge(g[["date","GPR_shock"]], on="date", how="inner"))

# ── 3. Build panel ────────────────────────────────────────────────────────────

print("Loading data …")
prod    = load_production()
quota_m = load_quota()
brent   = monthly_mean(BRENT_XLSX, "brent")
vix     = monthly_mean(VIX_XLSX,   "vix")
gpr_raw = load_gpr()
oil_gdp = load_oil_gdp()
inst    = load_inst()

print("Constructing shocks (Ready, 2018) …")
shocks = build_shocks(brent, vix, gpr_raw)
shocks = shocks.merge(vix, on="date", how="left")
print(f"  ORS ⊥ ODS verified  ✓")

# Merge production + quota → deviation
merged = prod.merge(quota_m, on=["iso3","date"], how="inner")
merged = merged[merged["quota"] >= merged["prod"] * 0.4]   # filter implausible quotas
merged["dev_pct"] = (merged["prod"] - merged["quota"]) / merged["quota"] * 100

panel = merged.merge(shocks, on="date", how="inner")
panel["year"] = panel["date"].dt.year
panel = (panel.merge(oil_gdp, on=["iso3","year"], how="left")
              .merge(inst,    on=["iso3","year"], how="left"))
panel = panel.sort_values(["iso3","date"]).reset_index(drop=True)
for col in ["oil_gdp","inst"]:
    panel[col] = panel.groupby("iso3")[col].ffill().bfill()
panel["ln_vix"] = np.log(panel["vix"])

# 12-month cumulative shocks + optimal lags
for shock in ["ORS","ODS","GPR_shock"]:
    panel[f"{shock}_12m"] = panel.groupby("iso3")[shock].transform(
        lambda x: x.rolling(12, min_periods=6).sum())
panel["ODS_l6"]     = panel.groupby("iso3")["ODS_12m"].shift(6)
panel["ORS_l6"]     = panel.groupby("iso3")["ORS_12m"].shift(6)
panel["GPR_l12"]    = panel.groupby("iso3")["GPR_shock_12m"].shift(12)
panel["ln_vix_12m"] = panel.groupby("iso3")["ln_vix"].transform(
    lambda x: x.rolling(12, min_periods=6).mean())
panel["dev_pct_l1"] = panel.groupby("iso3")["dev_pct"].shift(1)

# Country dummies (SAU = reference, omitted)
for iso3 in OPEC:
    if iso3 != REF_COUNTRY:
        panel[f"C_{iso3}"] = (panel["iso3"] == iso3).astype(float)

KEY = ["dev_pct","ODS_l6","ORS_l6","GPR_l12",
       "ln_vix_12m","oil_gdp","inst","dev_pct_l1"]
p = panel.dropna(subset=KEY).copy()

SHOCK_F   = ["ODS_l6","ORS_l6","GPR_l12"]
CTRL_F    = ["ln_vix_12m","oil_gdp","inst","dev_pct_l1"]
DUMMY_F   = [f"C_{c}" for c in OPEC if c != REF_COUNTRY]
ALL_F     = SHOCK_F + CTRL_F + DUMMY_F
N_CONT    = len(SHOCK_F) + len(CTRL_F)

# Sort chronologically (pooled, by date then country for consistency)
p = p.sort_values(["date","iso3"]).reset_index(drop=True)

X_raw = p[ALL_F].values.astype(float)
y     = p["dev_pct"].values.astype(float)
dates = p["date"].values

print(f"\nPanel: {len(p):,} obs | {p['iso3'].nunique()} countries | "
      f"{p['date'].min().date()} – {p['date'].max().date()}")
print(f"Features: {len(ALL_F)}  "
      f"({len(SHOCK_F)} shocks + {len(CTRL_F)} controls + "
      f"{len(DUMMY_F)} country dummies)")
print(f"DV: mean={y.mean():+.2f}%  std={y.std():.2f}%  "
      f"range [{y.min():.1f}%, {y.max():.1f}%]")
print(f"\nRolling CV: window={WINDOW_ROWS} rows (~6 yrs), "
      f"step={STEP_ROWS} rows (~1 yr)")

# ── 4. Rolling Time Series Cross-Validation ───────────────────────────────────

n = len(p)
fold_starts = range(0, n - WINDOW_ROWS - STEP_ROWS + 1, STEP_ROWS)
n_folds = sum(1 for s in fold_starts
              if s + WINDOW_ROWS < n)

print(f"Total folds: {n_folds}")
print(f"\n── Rolling CV with nested hyperparameter tuning ──")
print(f"   Inner CV: {N_INNER_SPLITS}-fold TimeSeriesSplit within each training window")
print(f"\n   {'Fold':>4}  {'Train':>22}  {'Test':>22}  "
      f"{'N_tr':>6}  {'N_te':>6}  {'R²':>8}  {'RMSE':>8} {'MAE':>8} {'Best C':>7}")
print("   " + "─"*85)

fold_results = []
all_oos_true = []
all_oos_pred = []
all_oos_idx  = []

for fold, start in enumerate(fold_starts, start=1):
    end_train  = start + WINDOW_ROWS
    start_test = end_train
    end_test   = min(start_test + STEP_ROWS, n)

    if start_test >= n:
        break

    tr_idx = np.arange(start, end_train)
    te_idx = np.arange(start_test, end_test)

    X_tr = X_raw[tr_idx].copy()
    X_te = X_raw[te_idx].copy()
    y_tr = y[tr_idx]
    y_te = y[te_idx]

    # Scale continuous features on training window only
    sc = StandardScaler()
    X_tr[:, :N_CONT] = sc.fit_transform(X_tr[:, :N_CONT])
    X_te[:, :N_CONT] = sc.transform(X_te[:, :N_CONT])

    # Inner CV: tune hyperparameters on training window
    inner_cv = TimeSeriesSplit(n_splits=N_INNER_SPLITS)
    gs = GridSearchCV(
        SVR(kernel="rbf"), PARAM_GRID,
        cv=inner_cv, scoring="r2",
        n_jobs=N_JOBS, refit=True, verbose=0,
    )
    gs.fit(X_tr, y_tr)

    y_pred = gs.best_estimator_.predict(X_te)
    r2     = r2_score(y_te, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    mae    = mean_absolute_error(y_te, y_pred)

    d_tr_from = pd.Timestamp(dates[tr_idx[0]]).strftime("%Y-%m")
    d_tr_to   = pd.Timestamp(dates[tr_idx[-1]]).strftime("%Y-%m")
    d_te_from = pd.Timestamp(dates[te_idx[0]]).strftime("%Y-%m")
    d_te_to   = pd.Timestamp(dates[te_idx[-1]]).strftime("%Y-%m")

    fold_results.append({
        "fold":      fold,
        "n_train":   len(tr_idx),
        "n_test":    len(te_idx),
        "r2_test":   r2,
        "rmse_test": rmse,
        "mae_test":  mae,
        "best_C":    gs.best_params_["C"],
        "best_eps":  gs.best_params_["epsilon"],
        "best_gamma":gs.best_params_["gamma"],
        "train_from":d_tr_from, "train_to":d_tr_to,
        "test_from": d_te_from, "test_to": d_te_to,
    })
    all_oos_true.extend(y_te)
    all_oos_pred.extend(y_pred)
    all_oos_idx.extend(te_idx.tolist())

    print(f"   {fold:>4}  {d_tr_from}–{d_tr_to}  {d_te_from}–{d_te_to}  "
          f"{len(tr_idx):>6}  {len(te_idx):>6}  "
          f"{r2:>+8.4f}  {rmse:>8.2f}pp  {gs.best_params_['C']:>7}")

fold_df  = pd.DataFrame(fold_results)
cv_r2    = fold_df["r2_test"].mean()
cv_rmse  = fold_df["rmse_test"].mean()
cv_mae   = fold_df["mae_test"].mean()
oos_r2   = r2_score(all_oos_true, all_oos_pred)
oos_rmse = np.sqrt(mean_squared_error(all_oos_true, all_oos_pred))
oos_mae  = mean_absolute_error(all_oos_true, all_oos_pred)

print(f"\n   Mean CV R²:         {cv_r2:+.4f}")
print(f"   Mean CV RMSE:       {cv_rmse:.2f} pp")
print(f"   Mean CV MAE:        {cv_mae:.2f} pp")
print(f"   OOS R² (pooled):    {oos_r2:+.4f}")
print(f"   OOS RMSE (pooled):  {oos_rmse:.2f} pp")
print(f"   OOS MAE (pooled):   {oos_mae:.2f} pp")

# ── 5. Final model on full data ───────────────────────────────────────────────

print(f"\n── Final model (full data) ──")

# Use modal best hyperparameters from CV folds
best_C   = fold_df["best_C"].mode()[0]
best_eps = fold_df["best_eps"].mode()[0]
best_gam = fold_df["best_gamma"].mode()[0]

sc_full = StandardScaler()
X_full  = X_raw.copy()
X_full[:, :N_CONT] = sc_full.fit_transform(X_raw[:, :N_CONT])

final_svr = SVR(kernel="rbf", C=best_C, epsilon=best_eps, gamma=best_gam)
final_svr.fit(X_full, y)
y_is_pred = final_svr.predict(X_full)
is_r2   = r2_score(y, y_is_pred)
is_rmse = np.sqrt(mean_squared_error(y, y_is_pred))
is_mae  = mean_absolute_error(y, y_is_pred)

print(f"   Hyperparameters: C={best_C}  ε={best_eps}  γ={best_gam}")
print(f"   In-sample R²:    {is_r2:.4f}")
print(f"   In-sample RMSE:  {is_rmse:.2f} pp")

# ── 6. Permutation feature importance (last fold OOS) ────────────────────────

print(f"\n── Permutation Importance ({N_PERM_REPS} repeats, last fold OOS) ──")

last_fold  = fold_results[-1]
last_start = last_fold["fold"] - 1
last_tr    = np.arange(
    (last_start) * STEP_ROWS,
    (last_start) * STEP_ROWS + WINDOW_ROWS)
last_te    = np.arange(
    (last_start) * STEP_ROWS + WINDOW_ROWS,
    min((last_start) * STEP_ROWS + WINDOW_ROWS + STEP_ROWS, n))

X_last_te = X_raw[last_te].copy()
sc_last   = StandardScaler()
X_raw[last_tr][:, :N_CONT]   # just to confirm
sc_last.fit(X_raw[last_tr][:, :N_CONT])
X_last_te[:, :N_CONT] = sc_last.transform(X_last_te[:, :N_CONT])

perm = permutation_importance(
    final_svr, X_last_te, y[last_te],
    n_repeats=N_PERM_REPS, random_state=42,
    scoring="r2", n_jobs=N_JOBS,
)
imp_mean = perm.importances_mean
imp_std  = perm.importances_std

print(f"   {'Feature':<22} {'Importance':>12} {'Std':>8}")
print("   " + "─"*44)
for fi in np.argsort(imp_mean)[::-1][:15]:
    bar = "█" * max(1, int(abs(imp_mean[fi]) /
                            (max(abs(imp_mean)) + 1e-9) * 20))
    print(f"   {ALL_F[fi]:<22} {imp_mean[fi]:>+12.5f} "
          f"{imp_std[fi]:>8.5f}  {bar}")

shock_imp = imp_mean[:len(SHOCK_F)].sum()
ctrl_imp  = imp_mean[len(SHOCK_F):N_CONT].sum()
dum_imp   = imp_mean[N_CONT:].sum()
print(f"\n   Group importance:")
print(f"     Shocks  (ODS+ORS+GPR): {shock_imp:+.5f}")
print(f"     Controls:               {ctrl_imp:+.5f}")
print(f"     Country dummies:        {dum_imp:+.5f}")

# ── 7. Results summary ────────────────────────────────────────────────────────

print(f"\n\n{'='*65}")
print("RESULTS SUMMARY — SVR RBF, Quota Deviation as DV")
print(f"{'='*65}")
print(f"DV:      dev_pct = (actual − quota) / quota × 100")
print(f"N:       {len(p):,} obs | {p['iso3'].nunique()} countries | "
      f"{p['date'].min().date()} – {p['date'].max().date()}")
print(f"Shocks:  ODS lag 6m | ORS lag 6m | GPR lag 12m (12m cumulative)")
print(f"Val:     Rolling TSCV | window={WINDOW_ROWS} rows | "
      f"step={STEP_ROWS} rows | {len(fold_results)} folds")
print(f"Tuning:  Nested inner {N_INNER_SPLITS}-fold CV | "
      f"grid: C={PARAM_GRID['C']}, ε={PARAM_GRID['epsilon']}")
print(f"\nFinal hyperparameters: C={best_C}  ε={best_eps}  γ={best_gam}")
print(f"\nPerformance:")
print(f"  In-sample  R²:   {is_r2:+.4f}")
print(f"  In-sample  RMSE: {is_rmse:.2f} pp")
print(f"  In-sample  MAE: {is_mae:.2f} pp")
print(f"  CV mean    R²:   {cv_r2:+.4f}")
print(f"  CV mean    RMSE: {cv_rmse:.2f} pp")
print(f"  OOS pooled R²:   {oos_r2:+.4f}")
print(f"  OOS pooled RMSE: {oos_rmse:.2f} pp")
print(f"  OOS pooled MAE:  {oos_mae:.2f} pp")
print(f"\nFold-level:")
print(f"  {'Fold':>4}  {'Test period':>22}  {'R²':>8}  {'RMSE':>8}   {'MAE':>8}   {'Best C':>7}")
print("  " + "─"*55)
for _, r in fold_df.iterrows():
    print(f"  {r['fold']:>4}  {r['test_from']}–{r['test_to']:>8}  "
          f"{r['r2_test']:>+8.4f}  {r['rmse_test']:>8.2f}pp {r['mae_test']:>8.2f}pp "
          f"{r['best_C']:>7}")

# ── 8. Figure: shock contributions to quota deviation over time ───────────────
#
# For each shock, compute its marginal contribution to the SVR prediction:
#   contribution_t = predict(X_full) − predict(X with shock zeroed out)
# Since features are standardised, zeroing = setting to 0 (the mean).
# Aggregated to monthly OPEC average across all countries.

contributions = {}
for shock in SHOCK_F:
    fi = ALL_F.index(shock)
    X_zeroed = X_full.copy()
    X_zeroed[:, fi] = 0.0
    contributions[shock] = y_is_pred - final_svr.predict(X_zeroed)

p_contrib = p[['date']].copy()
for shock in SHOCK_F:
    p_contrib[shock] = contributions[shock]
monthly = (p_contrib.groupby('date')[SHOCK_F].mean()
                    .reset_index()
                    .sort_values('date'))

SHOCK_COLORS = {
    'ODS_l6':  '#4575b4',
    'ORS_l6':  '#d73027',
    'GPR_l12': '#762a83',
}
SHOCK_LABELS = {
    'ODS_l6':  'Oil Demand Shock',
    'ORS_l6':  'Oil Risk Shock',
    'GPR_l12': 'GPR Shock',
}

fig, ax = plt.subplots(figsize=(14, 5))

for shock in SHOCK_F:
    smoothed = monthly[shock].rolling(3, center=True, min_periods=1).mean()
    ax.plot(monthly['date'].values, smoothed.values,
            color=SHOCK_COLORS[shock], linewidth=1.8, alpha=0.9)

ax.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.5)

ax.set_xlabel('Year', fontsize=12, labelpad=8)
ax.set_ylabel('Contribution to quota deviation (pp)', fontsize=12, labelpad=8)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=10)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{x:+.1f}'))
ax.tick_params(axis='y', labelsize=10)
ax.grid(axis='y', alpha=0.2, linewidth=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_svr_dev.png', dpi=180, bbox_inches='tight')

# ── 9. Figure: permutation importance ────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(10, 6))

colors = (["#4575b4"] * len(SHOCK_F) +
          ["#1a9850"] * len(CTRL_F) +
          ["#878787"] * len(DUMMY_F))

top_n     = 14
top_order = np.argsort(np.abs(imp_mean))[-top_n:]
y_pos     = np.arange(len(top_order))

ax2.barh(y_pos, imp_mean[top_order],
         xerr=imp_std[top_order],
         color=[colors[i] for i in top_order],
         alpha=0.85, height=0.6,
         error_kw={"elinewidth": 1.2, "capsize": 3})
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([ALL_F[i] for i in top_order], fontsize=10)
ax2.set_xlabel(
    "Mean R² decrease when feature is randomly shuffled", fontsize=10)
ax2.grid(axis="x", alpha=0.25)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

from matplotlib.patches import Patch
ax2.legend(handles=[
    Patch(facecolor="#4575b4", alpha=0.85, label="Shock variables"),
    Patch(facecolor="#1a9850", alpha=0.85, label="Control variables"),
    Patch(facecolor="#878787", alpha=0.85, label="Country dummies"),
], loc="lower right", fontsize=9, frameon=True)

plt.tight_layout()
plt.savefig("fig_svr_dev_importance.png", dpi=180, bbox_inches="tight")
print("Saved: fig_svr_dev_importance.png")

