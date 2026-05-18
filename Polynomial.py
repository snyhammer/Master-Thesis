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

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))

# For user: insert directory path here
PROD_CSV   = os.path.join(DATA_DIR, "COil_production_monthly.csv")
BRENT_XLSX = os.path.join(DATA_DIR, "DCOILBRENTEU.xlsx")
GPR_XLS    = os.path.join(DATA_DIR, "GPR.xls")
VIX_XLSX   = os.path.join(DATA_DIR, "VIXCLS.xlsx")
OILGDP_XLS = os.path.join(DATA_DIR, "Oil_%_of_GDP.xls")
INST_XLSX  = os.path.join(DATA_DIR, "Institutional_Quality_Index.xlsx")
QUOTA_XLSX = os.path.join(DATA_DIR, "OPEC_Quotas.xlsx")

OPEC = {
    "SAU":"Saudi Arabia","IRN":"Iran","IRQ":"Iraq","KWT":"Kuwait",
    "ARE":"UAE","VEN":"Venezuela","NGA":"Nigeria","DZA":"Algeria",
    "GAB":"Gabon","COG":"Congo","GNQ":"Eq. Guinea","LBY":"Libya",
}
OPEC_NAME_MAP = {
    "Algeria":"DZA","Congo":"COG","Equatorial Guinea":"GNQ","Gabon":"GAB",
    "IR Iran":"IRN","Iraq":"IRQ","Kuwait":"KWT","Libya":"LBY",
    "Nigeria":"NGA","Saudi Arabia":"SAU","United Arab Emirates":"ARE","Venezuela":"VEN",
}
REF_COUNTRY = "SAU"

WINDOW_ROWS = 616
N_SPLITS    = 5
CUM_WINDOW  = 6
LAG_SHOCKS  = 9
YOY_HORIZON = 12

SHOCK_F = ["ODS_L", "ORS_L", "GPR_L"]
CTRL_F  = ["oil_gdp", "inst", "d_pol_stab", "quota_tightness", "yoy_dev_l1", "regime"]

N_BOOTSTRAP  = 1000
BLOCK_LENGTH = 12
CI_ALPHA     = 0.05

NAVY  = "#1B2A6B"
ODS_C = "#e07b39"


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
        if not isinstance(row["Allocation_kbd"], (int, float, np.integer)): continue
        val = float(row["Allocation_kbd"])
        if np.isnan(val): continue
        iso3 = OPEC_NAME_MAP.get(row["Country"])
        if iso3 is None: continue
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
    df["date"] = (pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp())
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
        "Iraq":"IRQ","Kuwait":"KWT","Nigeria":"NGA","Libya":"LBY",
        "Saudi Arabia":"SAU","United Arab Emirates":"ARE","Venezuela":"VEN","Congo":"COG",
    }
    df["iso3"] = df["Country"].map(nm)
    df = df.dropna(subset=["iso3"])
    pol_col = [c for c in df.columns if "Political stability" in str(c)]
    wgi = [c for c in df.columns if any(k in c for k in
           ["effectiveness","corruption","Regulatory","Political stability"])]
    df["inst"]     = df[wgi].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    df["pol_stab"] = df[pol_col[0]].apply(pd.to_numeric, errors="coerce") if pol_col else np.nan
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df[["iso3","year","inst","pol_stab"]]

def build_shocks(brent, vix, gpr):
    df = (brent.merge(vix, on="date", how="inner").sort_values("date").reset_index(drop=True))
    df["d_ln_brent"] = np.log(df["brent"]).diff()
    df = df.dropna()
    arma = ARIMA(df["vix"].values, order=(1, 0, 1)).fit()
    df["ORS"] = arma.resid
    X = np.column_stack([np.ones(len(df)), df["ORS"].values])
    coef, *_ = np.linalg.lstsq(X, df["d_ln_brent"].values, rcond=None)
    df["ODS"] = df["d_ln_brent"].values - X @ coef
    assert abs(df["ORS"].corr(df["ODS"])) < 1e-6
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

def block_bootstrap_poly(X_poly, y, oos_true, oos_pred,
                          n_boot=N_BOOTSTRAP, block_len=BLOCK_LENGTH, alpha=CI_ALPHA):
    n     = len(y)
    n_oos = len(oos_true)
    n_blocks     = int(np.ceil(n     / block_len))
    n_blocks_oos = int(np.ceil(n_oos / block_len))
    boot_r2=[]; boot_rmse=[]; boot_mae=[]
    boot_r2_oos=[]; boot_rmse_oos=[]; boot_mae_oos=[]
    oos_true = np.array(oos_true); oos_pred = np.array(oos_pred)
    rng = np.random.default_rng(42)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
        idx    = np.concatenate([np.arange(s, min(s + block_len, n)) for s in starts])[:n]
        Xb = X_poly[idx].copy(); yb = y[idx]
        if len(np.unique(yb)) < 3: continue
        sc_b = StandardScaler()
        Xb[:, :N_CONT_POLY] = sc_b.fit_transform(Xb[:, :N_CONT_POLY])
        mdl_b  = LinearRegression().fit(Xb, yb)
        pred_b = mdl_b.predict(Xb)
        boot_r2.append(r2_score(yb, pred_b))
        boot_rmse.append(np.sqrt(mean_squared_error(yb, pred_b)))
        boot_mae.append(mean_absolute_error(yb, pred_b))
        starts_o = rng.integers(0, max(1, n_oos - block_len + 1), size=n_blocks_oos)
        idx_o    = np.concatenate([np.arange(s, min(s + block_len, n_oos)) for s in starts_o])[:n_oos]
        yb_o = oos_true[idx_o]; pb_o = oos_pred[idx_o]
        if len(np.unique(yb_o)) >= 3:
            boot_r2_oos.append(r2_score(yb_o, pb_o))
            boot_rmse_oos.append(np.sqrt(mean_squared_error(yb_o, pb_o)))
            boot_mae_oos.append(mean_absolute_error(yb_o, pb_o))
        if (b + 1) % 100 == 0: print(f"    {b+1}/{n_boot} done", flush=True)
    def ci(dist):
        dist = np.array(dist)
        return (np.percentile(dist, alpha/2*100), np.percentile(dist, (1-alpha/2)*100), dist)
    return {"r2":ci(boot_r2),"rmse":ci(boot_rmse),"mae":ci(boot_mae),
            "r2_oos":ci(boot_r2_oos),"rmse_oos":ci(boot_rmse_oos),"mae_oos":ci(boot_mae_oos)}

def plot_bootstrap_poly(point_ests, boot_res):
    configs = [
        ("r2",     "R²",        point_ests["is_r2"],   "In-sample R²"),
        ("rmse",   "RMSE (pp)", point_ests["is_rmse"], "In-sample RMSE"),
        ("mae",    "MAE (pp)",  point_ests["is_mae"],  "In-sample MAE"),
        ("r2_oos", "R²",        point_ests["oos_r2"],  "OOS R²"),
        ("rmse_oos","RMSE (pp)",point_ests["oos_rmse"],"OOS RMSE"),
        ("mae_oos", "MAE (pp)", point_ests["oos_mae"], "OOS MAE"),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(27, 5))
    for ax, (key, xlabel, pt, title) in zip(axes, configs):
        lo, hi, dist = boot_res[key]
        ax.hist(dist, bins=40, color=NAVY, alpha=0.70, edgecolor="white")
        ax.axvline(pt, color="#d73027", lw=2.5, ls="-", label=f"Estimate: {pt:.4f}")
        ax.axvline(lo, color=ODS_C, lw=1.8, ls="--", label=f"95% CI: [{lo:.4f}, {hi:.4f}]")
        ax.axvline(hi, color=ODS_C, lw=1.8, ls="--")
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel("Bootstrap frequency", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5, framealpha=0.9); ax.set_facecolor("white")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(f"Block Bootstrap 95% Confidence Intervals — Polynomial Regression Baseline\n"
                 f"(n_boot={N_BOOTSTRAP}, block_len={BLOCK_LENGTH}m  |  YoY Δ Deviation DV)",
                 fontsize=11, fontweight="bold")
    fig.patch.set_facecolor("white"); plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "bootstrap_poly_fig.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


print("Loading data...")
prod    = load_production()
quota_m = load_quota()
brent   = monthly_mean(BRENT_XLSX, "brent")
vix     = monthly_mean(VIX_XLSX,   "vix")
gpr_raw = load_gpr()
oil_gdp = load_oil_gdp()
inst    = load_inst()

print("Constructing shocks...")
shocks = build_shocks(brent, vix, gpr_raw)
shocks = shocks.merge(vix, on="date", how="left")

merged = prod.merge(quota_m, on=["iso3","date"], how="inner")
merged = merged[merged["quota"] >= merged["prod"] * 0.4]
merged["dev_pct"] = (merged["prod"] - merged["quota"]) / merged["quota"] * 100

panel = merged.merge(shocks, on="date", how="inner")
panel["year"] = panel["date"].dt.year
panel = (panel.merge(oil_gdp, on=["iso3","year"], how="left")
              .merge(inst,    on=["iso3","year"], how="left"))
panel = panel.sort_values(["iso3","date"]).reset_index(drop=True)
for col in ["oil_gdp","inst","pol_stab"]:
    panel[col] = panel.groupby("iso3")[col].ffill().bfill()
panel["ln_vix"] = np.log(panel["vix"])

panel["yoy_dev"]    = panel.groupby("iso3")["dev_pct"].transform(lambda x: x - x.shift(YOY_HORIZON))
panel["yoy_dev_l1"] = panel.groupby("iso3")["yoy_dev"].transform(lambda x: x.shift(1))

panel["quota_tightness"] = panel["quota"] / (
    panel.groupby("iso3")["prod"].transform(
        lambda x: x.rolling(12, min_periods=3).mean()) + 1e-9)

panel["d_pol_stab"] = panel.groupby("iso3")["pol_stab"].transform(lambda x: x.diff())

def get_regime(date):
    if date < pd.Timestamp("2009-01-01"): return 0
    elif date < pd.Timestamp("2017-01-01"): return 1
    else: return 2
panel["regime"] = panel["date"].apply(get_regime)

for shock in ["ORS","ODS","GPR_shock"]:
    panel[f"{shock}_cum"] = panel.groupby("iso3")[shock].transform(
        lambda x: x.rolling(CUM_WINDOW, min_periods=1).mean())
panel["ODS_L"] = panel.groupby("iso3")["ODS_cum"].transform(lambda x: x.shift(LAG_SHOCKS))
panel["ORS_L"] = panel.groupby("iso3")["ORS_cum"].transform(lambda x: x.shift(LAG_SHOCKS))
panel["GPR_L"] = panel.groupby("iso3")["GPR_shock_cum"].transform(lambda x: x.shift(LAG_SHOCKS))

for iso3 in OPEC:
    if iso3 != REF_COUNTRY:
        panel[f"C_{iso3}"] = (panel["iso3"] == iso3).astype(float)

DUMMY_F = [f"C_{c}" for c in OPEC if c != REF_COUNTRY]
ALL_F   = SHOCK_F + CTRL_F + DUMMY_F
N_CONT  = len(SHOCK_F) + len(CTRL_F)

KEY = ["yoy_dev"] + ALL_F
p = panel.dropna(subset=KEY).copy()
p = p.sort_values(["date","iso3"]).reset_index(drop=True)

X_raw  = p[ALL_F].values.astype(float)
y      = p["yoy_dev"].values.astype(float)
n      = len(p)
dates  = p["date"].values

shock_idx    = [ALL_F.index(s) for s in SHOCK_F]
X_sq         = X_raw[:, shock_idx] ** 2
X_poly       = np.hstack([X_raw, X_sq])
feat_names   = ALL_F + [f"{s}^2" for s in SHOCK_F]
N_CONT_POLY  = N_CONT + len(SHOCK_F)

tscv = TimeSeriesSplit(n_splits=N_SPLITS, max_train_size=WINDOW_ROWS)

print("Running rolling CV...")
fold_results = []
all_oos_true = []
all_oos_pred = []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_poly), start=1):
    X_tr = X_poly[tr_idx].copy(); X_te = X_poly[te_idx].copy()
    y_tr = y[tr_idx]; y_te = y[te_idx]
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
    fold_results.append({"fold":fold,"test_from":d_te_from,"test_to":d_te_to,
                         "n_train":len(tr_idx),"n_test":len(te_idx),
                         "r2_test":r2,"rmse_test":rmse,"mae_test":mae})
    all_oos_true.extend(y_te); all_oos_pred.extend(y_pred)

fold_df  = pd.DataFrame(fold_results)
cv_r2    = fold_df["r2_test"].mean()
cv_rmse  = fold_df["rmse_test"].mean()
cv_mae   = fold_df["mae_test"].mean()
oos_r2   = r2_score(all_oos_true, all_oos_pred)
oos_rmse = np.sqrt(mean_squared_error(all_oos_true, all_oos_pred))
oos_mae  = mean_absolute_error(all_oos_true, all_oos_pred)

sc_full = StandardScaler()
X_full  = X_poly.copy()
X_full[:, :N_CONT_POLY] = sc_full.fit_transform(X_poly[:, :N_CONT_POLY])
final   = LinearRegression().fit(X_full, y)
yis     = final.predict(X_full)
is_r2   = r2_score(y, yis)
is_rmse = np.sqrt(mean_squared_error(y, yis))
is_mae  = mean_absolute_error(y, yis)

print(f"  IS  R²: {is_r2:+.4f}   RMSE: {is_rmse:.2f} pp   MAE: {is_mae:.2f} pp")
print(f"  OOS R²: {oos_r2:+.4f}   RMSE: {oos_rmse:.2f} pp   MAE: {oos_mae:.2f} pp")

print("Running block bootstrap...")
boot_res = block_bootstrap_poly(X_poly, y, oos_true=all_oos_true, oos_pred=all_oos_pred)

r2_lo,  r2_hi,  _  = boot_res["r2"]
rm_lo,  rm_hi,  _  = boot_res["rmse"]
ma_lo,  ma_hi,  _  = boot_res["mae"]
r2_oos_lo,  r2_oos_hi,  _ = boot_res["r2_oos"]
rm_oos_lo,  rm_oos_hi,  _ = boot_res["rmse_oos"]
ma_oos_lo,  ma_oos_hi,  _ = boot_res["mae_oos"]

boot_rows = [
    {"metric":"R² (in-sample)",  "estimate":round(is_r2,4),   "ci_lo":round(r2_lo,4),     "ci_hi":round(r2_hi,4),     "unit":""},
    {"metric":"RMSE (in-sample)","estimate":round(is_rmse,4),  "ci_lo":round(rm_lo,4),     "ci_hi":round(rm_hi,4),     "unit":"pp"},
    {"metric":"MAE (in-sample)", "estimate":round(is_mae,4),   "ci_lo":round(ma_lo,4),     "ci_hi":round(ma_hi,4),     "unit":"pp"},
    {"metric":"R² (OOS)",        "estimate":round(oos_r2,4),   "ci_lo":round(r2_oos_lo,4), "ci_hi":round(r2_oos_hi,4), "unit":""},
    {"metric":"RMSE (OOS)",      "estimate":round(oos_rmse,4), "ci_lo":round(rm_oos_lo,4), "ci_hi":round(rm_oos_hi,4), "unit":"pp"},
    {"metric":"MAE (OOS)",       "estimate":round(oos_mae,4),  "ci_lo":round(ma_oos_lo,4), "ci_hi":round(ma_oos_hi,4), "unit":"pp"},
]
pd.DataFrame(boot_rows).to_csv(os.path.join(DATA_DIR, "bootstrap_poly_results.csv"), index=False)

point_ests_boot = {"is_r2":is_r2,"is_rmse":is_rmse,"is_mae":is_mae,
                   "oos_r2":oos_r2,"oos_rmse":oos_rmse,"oos_mae":oos_mae}
plot_bootstrap_poly(point_ests_boot, boot_res)

metrics   = ["R²", "RMSE (pp)", "MAE (pp)"]
is_vals   = [is_r2,  is_rmse,  is_mae]
oos_vals  = [oos_r2, oos_rmse, oos_mae]
colours   = {"In-sample": "#003153", "OOS pooled": "#c0392b"}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, metric, is_v, oos_v in zip(axes, metrics, is_vals, oos_vals):
    bars = ax.bar(list(colours.keys()), [is_v, oos_v], color=list(colours.values()), width=0.5, alpha=0.88)
    for bar, val in zip(bars, [is_v, oos_v]):
        sign = "+" if metric == "R²" and val >= 0 else ""
        fmt  = f"{sign}{val:.3f}" if metric == "R²" else f"{val:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + abs(bar.get_height()) * 0.02,
                fmt, ha="center", va="bottom", fontsize=11, fontweight="medium", color="#1a1a1a")
    ax.set_title(metric, fontsize=13, pad=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10); ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    if metric != "R²": ax.set_ylim(bottom=0)
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(DATA_DIR, "fig_poly_deltaDev_metrics.png"), dpi=180, bbox_inches="tight")

shock_terms = SHOCK_F + [f"{s}^2" for s in SHOCK_F]
label_map = {"ODS_L":"ODS  (linear)","ODS_L^2":"ODS  (squared)",
             "ORS_L":"ORS  (linear)","ORS_L^2":"ORS  (squared)",
             "GPR_L":"GPR  (linear)","GPR_L^2":"GPR  (squared)"}
colour_map = {"ODS_L":"#003153","ODS_L^2":"#6699bb","ORS_L":"#c0392b",
              "ORS_L^2":"#e07b72","GPR_L":"#1a7a4a","GPR_L^2":"#70bb8a"}
order = ["ODS_L","ODS_L^2","ORS_L","ORS_L^2","GPR_L","GPR_L^2"]
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
    ax2.text(xpos, bar.get_y() + bar.get_height()/2, f"{val:+.4f}",
             va="center", ha=ha, fontsize=9.5, color="#1a1a1a")
ax2.axvline(0, color="black", linewidth=0.9)
ax2.set_yticks(y_pos); ax2.set_yticklabels(labs, fontsize=11)
ax2.set_xlabel("Standardised coefficient — effect on quota deviation (pp)", fontsize=10, labelpad=8)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.grid(axis="x", alpha=0.2, linewidth=0.6)
from matplotlib.patches import Patch
ax2.legend(handles=[
    Patch(facecolor="#003153", alpha=0.88, label="Oil Demand Shock (ODS)"),
    Patch(facecolor="#c0392b", alpha=0.88, label="Oil Risk Shock (ORS)"),
    Patch(facecolor="#1a7a4a", alpha=0.88, label="Geopolitical Risk (GPR)"),
], loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(DATA_DIR, "fig_poly_deltaDev_coefs.png"), dpi=180, bbox_inches="tight")

rows = []
for _, r in fold_df.iterrows():
    rows.append({"type":"cv_fold","fold":r["fold"],"test_from":r["test_from"],"test_to":r["test_to"],
                 "n_train":r["n_train"],"n_test":r["n_test"],"r2":r["r2_test"],"rmse_pp":r["rmse_test"],"mae_pp":r["mae_test"]})
for label,rv,rmv,rmae in [("in_sample",is_r2,is_rmse,is_mae),("cv_mean",cv_r2,cv_rmse,cv_mae),("oos_pooled",oos_r2,oos_rmse,oos_mae)]:
    rows.append({"type":label,"r2":rv,"rmse_pp":rmv,"mae_pp":rmae})
for name,coef in zip(feat_names, final.coef_):
    rows.append({"type":"coefficient","feature":name,"coef_std":coef})
pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "results_poly_deltaDev.csv"), index=False)

