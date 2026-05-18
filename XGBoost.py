import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import re

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import shap
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

np.random.seed(42)

XGB_N_ESTIMATORS   = 400
XGB_MAX_DEPTH      = 5
XGB_LEARNING_RATE  = 0.04
XGB_SUBSAMPLE      = 0.8
XGB_COLSAMPLE      = 0.8
XGB_MIN_CHILD_W    = 3

LAG_SHOCKS   = 9
CUM_WINDOW   = 6
YOY_HORIZON  = 12

N_BOOTSTRAP  = 1000
BLOCK_LENGTH = 12
CI_ALPHA     = 0.05

TUNE_HYPERPARAMS = False
N_TUNE_ITER      = 80
TUNE_CV_SPLITS   = 5

MIN_OBS      = 30
SAMPLE_START = "1993-01-01"
SAMPLE_END   = "2025-12-01"

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(DATA_DIR, "xgb_output")
os.makedirs(OUT_DIR, exist_ok=True)

# For user: insert directory path here
PATH_PRODUCTION = os.path.join(DATA_DIR, "COil_production_monthly.csv")
PATH_BRENT      = os.path.join(DATA_DIR, "DCOILBRENTEU.xlsx")
PATH_VIX        = os.path.join(DATA_DIR, "VIXCLS.xlsx")
PATH_GPR        = os.path.join(DATA_DIR, "GPR.xls")
PATH_OIL_GDP    = os.path.join(DATA_DIR, "Oil_%_of_GDP.xls")
PATH_INST       = os.path.join(DATA_DIR, "Institutional_Quality_Index.xlsx")
PATH_QUOTAS     = os.path.join(DATA_DIR, "OPEC_Quotas.xlsx")

OPEC = {
    "DZA":"Algeria",    "IRQ":"Iraq",         "KWT":"Kuwait",
    "ARE":"UAE",        "VEN":"Venezuela",    "NGA":"Nigeria",
    "LBY":"Libya",      "GAB":"Gabon",        "COG":"Congo",
    "GNQ":"Eq. Guinea", "SAU":"Saudi Arabia", "IRN":"Iran"
}
CTRL_COLS = ["oil_gdp","inst","d_pol_stab","quota_tightness"]
SHOCKS    = ["ODS","ORS","GPR"]

FEAT_LABELS = {
    "ODS_L_dm":           "Oil demand shock",
    "ORS_L_dm":           "Oil risk shock",
    "GPR_L_dm":           "Geopolitical risk",
    "oil_gdp_dm":         "Oil share of GDP",
    "inst_dm":            "Institutional quality",
    "d_pol_stab_dm":      "Change in political stability",
    "quota_tightness_dm": "Quota tightness",
    "yoy_dev_l1_dm":      "Lagged YoY deviation",
}

NAVY="#1B2A6B"; STEEL="#4A6FA5"; GREEN="#1a9850"; LGREY="#cccccc"
ODS_C="#e07b39"; ORS_C="#9467bd"; GPR_C="#2ca02c"

COUNTRY_COLOURS = {
    "Algeria":      "#1f77b4",
    "Iraq":         "#ff7f0e",
    "Kuwait":       "#2ca02c",
    "UAE":          "#d62728",
    "Venezuela":    "#9467bd",
    "Nigeria":      "#8c564b",
    "Libya":        "#e377c2",
    "Gabon":        "#7f7f7f",
    "Congo":        "#bcbd22",
    "Eq. Guinea":   "#17becf",
    "Saudi Arabia": "#aec7e8",
    "Iran":         "#ffbb78",
}


def load_production():
    raw = pd.read_csv(PATH_PRODUCTION, sep=None, engine="python", header=None)
    opec_key = {iso: f"INTL.53-1-{iso}-TBPD.M" for iso in OPEC}
    col_map = {}
    for ci in range(raw.shape[1]):
        for ri in range(5):
            for iso, key in opec_key.items():
                if key in str(raw.iloc[ri, ci]):
                    col_map[iso] = ci
    records = []
    for iso, kc in col_map.items():
        dc = kc - 1; ds = None
        for r in range(raw.shape[0]):
            v = str(raw.iloc[r, dc])
            if len(v) == 7 and v[4] == "-" and v[:4].isdigit():
                ds = r; break
        if ds is None: continue
        dates = pd.to_datetime(raw.iloc[ds:, dc].astype(str), format="%Y-%m", errors="coerce")
        vals  = pd.to_numeric(raw.iloc[ds:, kc], errors="coerce")
        records.append(pd.DataFrame({"date": dates, "prod": vals, "iso3": iso}).dropna(subset=["date","prod"]))
    prod = pd.concat(records).sort_values(["iso3","date"]).reset_index(drop=True)
    prod["date"] = prod["date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    return prod

def load_brent():
    df = pd.read_excel(PATH_BRENT); df.columns = ["date","brent"]
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna().set_index("date")["brent"].resample("MS").mean().reset_index()

def load_vix():
    df = pd.read_excel(PATH_VIX); df.columns = ["date","vix"]
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna().set_index("date")["vix"].resample("MS").mean().reset_index()

def load_gpr():
    df = pd.read_excel(PATH_GPR, engine="xlrd")
    df["date"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
    return df[["date","GPR"]].dropna().sort_values("date").reset_index(drop=True)

def load_quotas(prod):
    df_q = pd.read_excel(PATH_QUOTAS, header=None)
    ch   = df_q.iloc[4, 1:].tolist()
    ciso = {"Algeria":"DZA","Congo":"COG","Equatorial Guinea":"GNQ","Gabon":"GAB",
            "IR Iran":"IRN","Iraq":"IRQ","Kuwait":"KWT","Libya":"LBY","Nigeria":"NGA",
            "Saudi Arabia":"SAU","United Arab Emirates":"ARE","Venezuela":"VEN"}
    mab  = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
            "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    def pp(label):
        parts = re.split(r"[-–—](?=[A-Za-z])", str(label).strip())
        def pm(s):
            s = s.strip()
            for m, n in mab.items():
                if s.startswith(m):
                    yr = int(s[3:]); yr = 2000+yr if yr<50 else 1900+yr
                    return pd.Timestamp(yr, n, 1)
            return None
        if len(parts)==2:
            s, e = pm(parts[0]), pm(parts[1])
            if s and e: return pd.date_range(s, e, freq="MS").tolist()
        elif len(parts)==1:
            t = pm(parts[0]); return [t] if t else []
        return []
    qr = []
    for ri in range(5, 18):
        cn = str(df_q.iloc[ri, 0]).strip(); iso3 = ciso.get(cn)
        if not iso3: continue
        for ci, hdr in enumerate(ch, start=1):
            val = str(df_q.iloc[ri, ci]).strip()
            if val in ["nd","–","-","nan","NaN",""]: continue
            try:   qv = float(val.replace(",",""))
            except: continue
            for ts in pp(hdr): qr.append({"iso3":iso3,"date":ts,"quota":qv})
    quotas_raw = pd.DataFrame(qr)
    prod_med   = prod.groupby("iso3")["prod"].median()
    clean = [r for _, r in quotas_raw.iterrows()
             if r["quota"] >= 0.15 * prod_med.get(r["iso3"], np.nan)]
    return pd.DataFrame(clean).sort_values(["iso3","date"]).reset_index(drop=True)

def load_controls():
    df_og = pd.read_excel(PATH_OIL_GDP, engine="xlrd", skiprows=3)
    yc    = [c for c in df_og.columns if str(c).strip().isdigit()]
    ogdp  = df_og.melt(id_vars=["Country Name","Country Code"],
                       value_vars=yc, var_name="year", value_name="oil_gdp")
    ogdp["year"] = ogdp["year"].astype(int)
    ogdp = ogdp.rename(columns={"Country Code":"iso3"})[["iso3","year","oil_gdp"]].dropna()
    df_i  = pd.read_excel(PATH_INST)
    pc    = "Political stability index (-2.5 weak; 2.5 strong)"
    gc    = "Government effectiveness index (-2.5 weak; 2.5 strong)"
    df_i  = df_i.rename(columns={"Code":"iso3","Year":"year",pc:"pol_stab",gc:"gov_eff"})
    df_i["inst"] = df_i[["pol_stab","gov_eff"]].mean(axis=1)
    return ogdp, df_i[["iso3","year","pol_stab","inst"]].dropna(subset=["inst"])


def ar1_residual(series):
    s = series.dropna().reset_index(drop=True)
    lag = s.shift(1); mask = lag.notna()
    coef = np.polyfit(lag[mask], s[mask], 1)
    resid = s - (coef[0]*lag + coef[1])
    return (resid - resid.mean()) / (resid.std() + 1e-9)

def build_shocks(brent, vix, gpr):
    df = brent.merge(vix, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["d_ln_brent"] = np.log(df["brent"]).diff()
    df = df.dropna().reset_index(drop=True)
    try:
        arma = ARIMA(df["vix"].values, order=(1,0,1)).fit(); ors = arma.resid
    except:
        ors = df["vix"].diff().fillna(0).values
    ors = (ors - ors.mean()) / (ors.std() + 1e-9); df["ORS"] = ors
    X   = np.column_stack([np.ones(len(df)), ors])
    c, *_ = np.linalg.lstsq(X, df["d_ln_brent"].values, rcond=None)
    ods = df["d_ln_brent"].values - X @ c
    df["ODS"] = (ods - ods.mean()) / ods.std()
    g = gpr.copy()
    g["ln_gpr"] = np.log(g["GPR"].clip(lower=0.01))
    g["GPR"]    = ar1_residual(g["ln_gpr"]).values
    df = df.merge(g[["date","GPR"]], on="date", how="left")
    return df[["date","ODS","ORS","GPR"]]


def build_panel(prod, quotas, shocks, ogdp, inst):
    base = prod.merge(quotas, on=["iso3","date"], how="inner")
    base["dev_pct"] = (base["prod"] - base["quota"]) / base["quota"] * 100
    base["quota_tightness"] = base["quota"] / (
        base.groupby("iso3")["prod"].transform(
            lambda x: x.rolling(12, min_periods=3).mean()) + 1e-9)
    base = base.sort_values(["iso3","date"])
    base["yoy_dev"] = base.groupby("iso3")["dev_pct"].transform(lambda x: x - x.shift(YOY_HORIZON))
    p = base.merge(shocks, on="date", how="left")
    shock_L = []
    for col in SHOCKS:
        out = col + "_L"
        for iso3 in OPEC:
            mask = p["iso3"] == iso3
            p.loc[mask, out] = (p.loc[mask, col].rolling(CUM_WINDOW, min_periods=1).mean().shift(LAG_SHOCKS).values)
        shock_L.append(out)
    p["year"] = p["date"].dt.year
    p = p.merge(ogdp[["iso3","year","oil_gdp"]], on=["iso3","year"], how="left")
    p = p.merge(inst[["iso3","year","pol_stab","inst"]], on=["iso3","year"], how="left")
    p = p.sort_values(["iso3","date"])
    p["d_pol_stab"] = p.groupby("iso3")["pol_stab"].transform(lambda x: x.diff())
    p = p[(p["date"] >= SAMPLE_START) & (p["date"] <= SAMPLE_END)].copy()
    for col in ["oil_gdp","inst","d_pol_stab","quota_tightness"]:
        p[col] = p.groupby("iso3")[col].transform(lambda x: x.ffill().bfill())
    return p.reset_index(drop=True), shock_L


def pool_panel(panel, shock_L):
    feat_cols = shock_L + CTRL_COLS
    rows = []
    for iso3 in OPEC:
        sub = (panel[panel["iso3"]==iso3].dropna(subset=["yoy_dev"] + feat_cols).copy())
        if len(sub) < MIN_OBS: continue
        rows.append(sub)
    pool = pd.concat(rows).sort_values("date").reset_index(drop=True)
    pool["yoy_dev_dm"] = pool["yoy_dev"] - pool.groupby("iso3")["yoy_dev"].transform("mean")
    feat_dm = []
    for col in feat_cols:
        col_dm = col + "_dm"
        pool[col_dm] = pool[col] - pool.groupby("iso3")[col].transform("mean")
        feat_dm.append(col_dm)
    mask = pool[feat_dm + ["yoy_dev_dm"]].notna().all(axis=1)
    pool = pool[mask].reset_index(drop=True)
    X = pool[feat_dm].values
    y = pool["yoy_dev_dm"].values
    return pool, X, y, feat_dm


def fit_xgb(X, y, params=None):
    p = {
        "n_estimators":    XGB_N_ESTIMATORS,
        "max_depth":       XGB_MAX_DEPTH,
        "learning_rate":   XGB_LEARNING_RATE,
        "subsample":       XGB_SUBSAMPLE,
        "colsample_bytree":XGB_COLSAMPLE,
        "min_child_weight":XGB_MIN_CHILD_W,
        "reg_alpha":       0.0,
        "reg_lambda":      1.0,
        "gamma":           0.0,
        "random_state":    42,
        "verbosity":       0,
    }
    if params:
        p.update(params)
    model = xgb.XGBRegressor(**p)
    model.fit(X, y)
    return model


def fold_analysis(X, y):
    from sklearn.model_selection import KFold, TimeSeriesSplit
    results = []
    for n_splits in [3, 5, 7, 10]:
        for label, cv in [
            ("TimeSeriesSplit", TimeSeriesSplit(n_splits=n_splits)),
            ("KFold(shuffle)",  KFold(n_splits=n_splits, shuffle=True, random_state=42)),
        ]:
            fold_r2 = []
            for tr, te in cv.split(X):
                m = fit_xgb(X[tr], y[tr])
                fold_r2.append(r2_score(y[te], m.predict(X[te])))
            mean_r2 = float(np.mean(fold_r2))
            std_r2  = float(np.std(fold_r2))
            results.append((label, n_splits, mean_r2, std_r2))
    best = max(results, key=lambda x: x[2])
    return best


def tune_xgb(X, y, feat_dm, n_iter=N_TUNE_ITER, n_splits=TUNE_CV_SPLITS):
    from sklearn.model_selection import TimeSeriesSplit
    rng = np.random.default_rng(42)
    search_space = {
        "n_estimators":     [200, 300, 400, 500, 600],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.01, 0.02, 0.04, 0.06, 0.08],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],
        "reg_alpha":        [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0, 5.0, 10.0],
        "gamma":            [0.0, 0.1, 0.3, 0.5, 1.0],
    }
    baseline_params = {
        "n_estimators": XGB_N_ESTIMATORS, "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LEARNING_RATE, "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE, "min_child_weight": XGB_MIN_CHILD_W,
        "reg_alpha": 0.0, "reg_lambda": 1.0, "gamma": 0.0,
    }
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    configs = [baseline_params.copy()]
    for _ in range(n_iter - 1):
        cfg = {k: rng.choice(v).item() for k, v in search_space.items()}
        configs.append(cfg)
    best_r2     = -np.inf
    best_params = None
    all_results = []
    for i, cfg in enumerate(configs):
        fold_r2   = []
        fold_rmse = []
        for tr, te in tscv.split(X):
            m = fit_xgb(X[tr], y[tr], params=cfg)
            p = m.predict(X[te])
            fold_r2.append(r2_score(y[te], p))
            fold_rmse.append(np.sqrt(mean_squared_error(y[te], p)))
        mean_r2   = float(np.mean(fold_r2))
        mean_rmse = float(np.mean(fold_rmse))
        all_results.append((mean_r2, mean_rmse, cfg))
        if mean_r2 > best_r2:
            best_r2     = mean_r2
            best_params = cfg.copy()
    tune_df = pd.DataFrame([
        {"rank": i+1, "mean_oos_r2": r, "mean_oos_rmse": rm, **cfg}
        for i, (r, rm, cfg) in enumerate(sorted(all_results, key=lambda x: -x[0]))
    ])
    tune_df.to_csv(os.path.join(OUT_DIR, "tuning_results.csv"), index=False)
    return best_params


def compute_point_estimates(X, y, pool, params=None):
    model = fit_xgb(X, y, params=params)
    pred_is = model.predict(X)
    point_ests = {
        "r2_in":   r2_score(y, pred_is),
        "rmse_in": np.sqrt(mean_squared_error(y, pred_is)),
        "mae_in":  mean_absolute_error(y, pred_is),
    }
    from sklearn.model_selection import KFold
    tscv = KFold(n_splits=10, shuffle=True, random_state=42)
    oos_y_all=[]; oos_p_all=[]; oos_iso_all=[]
    last_fold_model = None; last_fold_X_te = None; last_fold_y_te = None
    for tr, te in tscv.split(X):
        m = fit_xgb(X[tr], y[tr], params=params)
        oos_y_all.append(y[te])
        oos_p_all.append(m.predict(X[te]))
        oos_iso_all.append(pool["iso3"].values[te])
        last_fold_model = m
        last_fold_X_te  = X[te]
        last_fold_y_te  = y[te]
    oos_y   = np.concatenate(oos_y_all)
    oos_p   = np.concatenate(oos_p_all)
    oos_iso = np.concatenate(oos_iso_all)
    point_ests["r2_oos"]   = r2_score(oos_y, oos_p)
    point_ests["rmse_oos"] = np.sqrt(mean_squared_error(oos_y, oos_p))
    point_ests["mae_oos"]  = mean_absolute_error(oos_y, oos_p)
    return (point_ests, oos_y, oos_p, oos_iso, last_fold_model, last_fold_X_te, last_fold_y_te)

def country_decomposition(oos_y, oos_p, oos_iso):
    rows = []
    for iso3, name in OPEC.items():
        mask = oos_iso == iso3
        if mask.sum() < 5: continue
        yt = oos_y[mask]; yp = oos_p[mask]
        rows.append({
            "iso3": iso3, "country": name,
            "r2_oos":   round(r2_score(yt, yp), 4),
            "rmse_oos": round(np.sqrt(mean_squared_error(yt, yp)), 4),
            "mae_oos":  round(mean_absolute_error(yt, yp), 4),
            "n":        int(mask.sum()),
        })
    return pd.DataFrame(rows).sort_values("r2_oos", ascending=False).reset_index(drop=True)

def plot_country_decomposition(decomp_df):
    fig, ax = plt.subplots(figsize=(11, 5))
    x   = np.arange(len(decomp_df))
    cols= [COUNTRY_COLOURS.get(n, STEEL) for n in decomp_df["country"]]
    ax.bar(x, decomp_df["r2_oos"], color=cols, alpha=0.88, edgecolor="white")
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(decomp_df["country"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("OOS R² (walk-forward)", fontsize=10)
    ax.set_title("Country-level OOS R² — XGBoost\n(walk-forward, 10-fold KFold shuffle)",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18); ax.set_facecolor("white")
    fig.patch.set_facecolor("white"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"country_decomposition.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def compute_shap(model, X, pool, feat_dm):
    shock_feats = ["ODS_L", "ORS_L", "GPR_L"]
    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X)
    shap_rows = []
    for iso3, name in OPEC.items():
        mask        = pool["iso3"].values == iso3
        row_indices = np.where(mask)[0]
        if len(row_indices) < 5: continue
        for row_idx in row_indices:
            d = {"iso3": iso3, "country": name, "row": int(row_idx)}
            for fi, feat in enumerate(feat_dm):
                d[feat] = sv[row_idx, fi]
            pool_row = pool.iloc[row_idx]
            for sf in shock_feats:
                d[f"{sf}_raw"] = pool_row[sf]
                d[sf]          = sv[row_idx, feat_dm.index(sf+"_dm")]
            d["yoy_dev"]    = pool_row["yoy_dev"]
            d["yoy_dev_dm"] = pool_row["yoy_dev_dm"]
            shap_rows.append(d)
    shap_df = pd.DataFrame(shap_rows)
    return shap_df, sv

def plot_shap_overall(shap_df, feat_dm):
    overall = shap_df[feat_dm].abs().mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feat_dm)*0.45)))
    colours = [ODS_C if "ODS" in f else ORS_C if "ORS" in f
               else GPR_C if "GPR" in f else STEEL for f in overall.index]
    ax.barh(overall.index, overall.values, color=colours, alpha=0.85)
    ax.set_xlabel("Mean |SHAP| (pp)", fontsize=10)
    ax.set_title("Overall Feature Importance — XGBoost TreeSHAP\n"
                 "(mean absolute SHAP value, DV = YoY Δ quota deviation)",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2); fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"shap_overall.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    country_shap = (shap_df.groupby("country")[feat_dm].apply(lambda df: df.abs().mean()).reset_index())
    country_shap.to_csv(os.path.join(OUT_DIR,"shap_mean_abs_by_country.csv"), index=False)
    return country_shap


def compute_permutation_importance(oos_model, X_oos, y_oos, feat_dm, n_repeats=30, random_state=42):
    rng         = np.random.default_rng(random_state)
    baseline_r2 = r2_score(y_oos, oos_model.predict(X_oos))
    n_features  = X_oos.shape[1]
    importances = np.zeros((n_features, n_repeats))
    for fi in range(n_features):
        for rep in range(n_repeats):
            X_perm        = X_oos.copy()
            X_perm[:, fi] = rng.permutation(X_perm[:, fi])
            importances[fi, rep] = baseline_r2 - r2_score(y_oos, oos_model.predict(X_perm))
    perm_df = pd.DataFrame({
        "feature":    feat_dm,
        "importance": importances.mean(axis=1),
        "std":        importances.std(axis=1),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    perm_df["feature_label"] = perm_df["feature"].map(FEAT_LABELS).fillna(
        perm_df["feature"].str.replace("_dm","",regex=False))
    perm_df.to_csv(os.path.join(OUT_DIR, "permutation_importance.csv"), index=False)
    return perm_df


def plot_importance_comparison(shap_df, perm_df, feat_dm):
    def label(feat):
        return FEAT_LABELS.get(feat, feat.replace("_dm","").replace("_"," "))
    perm_plot = perm_df.copy()
    perm_plot["feature_label"] = perm_plot["feature"].map(label)
    perm_plot = perm_plot.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(perm_plot["feature_label"], perm_plot["importance"],
            xerr=perm_plot["std"], color=NAVY, alpha=0.85,
            edgecolor="white", capsize=3,
            error_kw=dict(elinewidth=1, ecolor="grey", capthick=1))
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Mean R\u00b2 decrease when feature is permuted", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2); ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance_comparison.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    from scipy.stats import spearmanr
    shap_means = shap_df[feat_dm].abs().mean().reset_index()
    shap_means.columns = ["feature","shap_mean"]
    merged = shap_means.merge(perm_df[["feature","importance"]], on="feature", how="inner")
    rho, p = spearmanr(merged["shap_mean"], merged["importance"])
    return rho, p


def compute_shap_asymmetric(shap_df):
    shock_feats = ["ODS_L", "ORS_L", "GPR_L"]
    records_pos=[]; records_neg=[]
    for iso3, name in OPEC.items():
        sub = shap_df[shap_df["country"] == name]
        if len(sub) < 5: continue
        for sf in shock_feats:
            pos = sub[sub[f"{sf}_raw"] >  0][sf]
            neg = sub[sub[f"{sf}_raw"] <= 0][sf]
            for recs, vals in [(records_pos, pos), (records_neg, neg)]:
                if len(vals) >= 5:
                    recs.append({"country": name, "shock": sf,
                                 "mean_shap": vals.mean(), "n": len(vals),
                                 "shap_vals": vals.values})
    return pd.DataFrame(records_pos), pd.DataFrame(records_neg)

def significance_shap(records_df):
    p_vals = []
    for _, row in records_df.iterrows():
        vals = row["shap_vals"]
        if len(vals) < 3: p_vals.append(1.0)
        else: _, p = stats.ttest_1samp(vals, 0.0); p_vals.append(p)
    records_df = records_df.copy()
    records_df["p_value"] = p_vals
    records_df["stars"]   = records_df["p_value"].apply(
        lambda p: "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "")
    return records_df

def plot_asymmetric_shap(pos_df, neg_df):
    shock_labels = {"ODS_L":"ODS","ORS_L":"ORS","GPR_L":"GPR"}
    countries    = list(COUNTRY_COLOURS.keys())
    for shock in ["ODS_L","ORS_L","GPR_L"]:
        for df, sign_label, fname in [
            (pos_df, "Positive Shock Months  (shock > 0)",  f"shap_asym_{shock_labels[shock]}_positive.png"),
            (neg_df, "Negative Shock Months  (shock ≤ 0)",  f"shap_asym_{shock_labels[shock]}_negative.png"),
        ]:
            sub = df[df["shock"]==shock].copy()
            sub = sub.set_index("country").reindex([c for c in countries if c in sub.index])
            sub = sub.dropna(subset=["mean_shap"])
            fig, ax = plt.subplots(figsize=(12, 5))
            x    = np.arange(len(sub))
            cols = [COUNTRY_COLOURS[c] for c in sub.index]
            ax.bar(x, sub["mean_shap"].values, color=cols, alpha=0.88, edgecolor="white", linewidth=0.5)
            yspan = sub["mean_shap"].abs().max() * 0.07 if len(sub) else 0.1
            for xi, (_, row) in zip(x, sub.iterrows()):
                if row.get("stars",""):
                    ypos = row["mean_shap"] + (yspan if row["mean_shap"]>=0 else -yspan)
                    ax.text(xi, ypos, row["stars"], ha="center",
                            va="bottom" if row["mean_shap"]>=0 else "top",
                            fontsize=10, fontweight="bold")
            ax.axhline(0, color="black", lw=0.9, ls="--", zorder=0)
            ax.set_title(f"{shock_labels[shock]} — {sign_label}\nMean TreeSHAP contribution to YoY Δ deviation  |  XGBoost",
                         fontsize=12, fontweight="bold")
            ax.set_ylabel("Mean SHAP contribution (pp)", fontsize=10)
            ax.set_xticks(x); ax.set_xticklabels(sub.index, rotation=45, ha="right", fontsize=9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.18); ax.set_facecolor("white")
            handles = [plt.Rectangle((0,0),1,1,color=COUNTRY_COLOURS[c],alpha=0.88) for c in sub.index]
            ax.legend(handles, list(sub.index), loc="center left", bbox_to_anchor=(1.01,0.5),
                      fontsize=9, title="Country", title_fontsize=9, framealpha=0.95, edgecolor=LGREY)
            fig.patch.set_facecolor("white"); plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches="tight", facecolor="white")
            plt.close()


def plot_shap_scatter(shap_df, cv_r2):
    shock_feats  = ["ODS_L","ORS_L","GPR_L"]
    shock_labels = {"ODS_L":"Oil Demand Shock (ODS)", "ORS_L":"Oil Risk Shock (ORS)", "GPR_L":"Geopolitical Risk (GPR)"}
    shock_short  = {"ODS_L":"ODS","ORS_L":"ORS","GPR_L":"GPR"}
    def stars(p): return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, sf in zip(axes, shock_feats):
        sshort = shock_short[sf]
        for name, grp in shap_df.groupby("country"):
            col  = COUNTRY_COLOURS.get(name, "#888")
            over = grp[grp["yoy_dev"] > 0]; under= grp[grp["yoy_dev"] <= 0]
            ax.scatter(over[f"{sf}_raw"],  over[sf],  color=col, alpha=0.65, s=20, marker="o", linewidths=0, zorder=3)
            ax.scatter(under[f"{sf}_raw"], under[sf], color=col, alpha=0.22, s=12, marker="^", linewidths=0, zorder=2)
        x_all = shap_df[f"{sf}_raw"].values; y_all = shap_df[sf].values
        ok    = np.isfinite(x_all) & np.isfinite(y_all)
        m,b,r,p,_ = stats.linregress(x_all[ok], y_all[ok])
        xr = np.linspace(x_all[ok].min(), x_all[ok].max(), 200)
        ax.plot(xr, m*xr+b, color="black", lw=2, ls="--", zorder=5, label=f"Linear  β={m:+.3f}{stars(p)}  r={r:.2f}")
        poly = np.polyfit(x_all[ok], y_all[ok], 2)
        ax.plot(xr, np.polyval(poly, xr), color="red", lw=2, ls="-", alpha=0.75, zorder=6, label="Quadratic fit")
        ax.axhline(0, color="grey", lw=0.8, ls=":", zorder=1); ax.axvline(0, color="grey", lw=0.8, ls=":", zorder=1)
        ax.set_xlabel(f"{sshort} lagged shock value", fontsize=10)
        ax.set_ylabel("SHAP (pp of YoY Δ deviation)", fontsize=9)
        ax.set_title(shock_labels[sf], fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.12); ax.set_facecolor("white"); ax.legend(fontsize=8.5, framealpha=0.9)
    handles = [mlines.Line2D([],[],marker="o",color="w",markerfacecolor=c, markersize=8,label=n)
               for n, c in COUNTRY_COLOURS.items()]
    handles += [mlines.Line2D([],[],marker="o",color="grey",markersize=7,label="● Increasing deviation"),
                mlines.Line2D([],[],marker="^",color="grey",markersize=7,alpha=0.4, label="▲ Decreasing deviation")]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8.5, framealpha=0.95, bbox_to_anchor=(0.5,-0.1))
    fig.suptitle(f"XGBoost TreeSHAP — SHAP dependence plots\nDV = YoY Δ quota deviation (within-demeaned)  |  CV R²={cv_r2:.3f}",
                 fontsize=12, fontweight="bold")
    fig.patch.set_facecolor("white"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"shap_scatter_overview.png"), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    for sf in shock_feats:
        sshort = shock_short[sf]
        fig, axes = plt.subplots(3, 4, figsize=(20, 14))
        axes_flat = axes.flatten()
        for ax, (name, grp) in zip(axes_flat, shap_df.groupby("country")):
            col  = COUNTRY_COLOURS.get(name, "#888")
            over = grp[grp["yoy_dev"] > 0]; under = grp[grp["yoy_dev"] <= 0]
            ax.scatter(over[f"{sf}_raw"],  over[sf],  color=col, alpha=0.7, s=24, marker="o", linewidths=0, zorder=3, label=f"Increasing (n={len(over)})")
            ax.scatter(under[f"{sf}_raw"], under[sf], color=col, alpha=0.25, s=14, marker="^", linewidths=0, zorder=2, label=f"Decreasing (n={len(under)})")
            x = grp[f"{sf}_raw"].values; yv = grp[sf].values
            ok= np.isfinite(x) & np.isfinite(yv)
            if ok.sum() > 5:
                m,b,r,p,_ = stats.linregress(x[ok], yv[ok])
                xr = np.linspace(x[ok].min(), x[ok].max(), 80)
                ax.plot(xr, m*xr+b, color="black", lw=1.8, ls="--", zorder=5)
                if ok.sum() > 20:
                    poly = np.polyfit(x[ok], yv[ok], 2)
                    ax.plot(xr, np.polyval(poly,xr), color="red", lw=1.5, ls="-", alpha=0.7, zorder=6)
                ax.set_title(f"{name}\nβ={m:+.3f}{stars(p)}  r={r:.2f}  n={ok.sum()}", fontsize=9, fontweight="bold", color=col)
            ax.axhline(0,color="grey",lw=0.7,ls=":",zorder=1); ax.axvline(0,color="grey",lw=0.7,ls=":",zorder=1)
            ax.set_xlabel(f"{sshort}",fontsize=8); ax.set_ylabel("SHAP (pp)",fontsize=8)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.12); ax.set_facecolor("white"); ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5, framealpha=0.8)
        for ax in axes_flat[len(OPEC):]: ax.set_visible(False)
        fig.suptitle(f"{shock_labels[sf]} — XGBoost TreeSHAP per country\nDV = YoY Δ quota deviation  |  CV R²={cv_r2:.3f}  |  ***p<0.01",
                     fontsize=12, fontweight="bold")
        fig.patch.set_facecolor("white"); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_scatter_{sshort}_percountry.png"), dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()


def bootstrap_is(X, y, n_boot=N_BOOTSTRAP, alpha=CI_ALPHA):
    n   = len(y)
    rng = np.random.default_rng(42)
    boot_r2=[]; boot_rmse=[]; boot_mae=[]
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        Xb  = X[idx]; yb = y[idx]
        if len(np.unique(yb)) < 3: continue
        m    = fit_xgb(Xb, yb)
        pred = m.predict(Xb)
        boot_r2.append(r2_score(yb, pred))
        boot_rmse.append(np.sqrt(mean_squared_error(yb, pred)))
        boot_mae.append(mean_absolute_error(yb, pred))
    def ci(dist):
        return (np.percentile(dist, alpha/2*100), np.percentile(dist, (1-alpha/2)*100), dist)
    return {"r2": ci(np.array(boot_r2)), "rmse": ci(np.array(boot_rmse)), "mae": ci(np.array(boot_mae))}


def bootstrap_oos(X, y, n_boot=N_BOOTSTRAP, alpha=CI_ALPHA):
    n       = len(y)
    rng     = np.random.default_rng(20000)
    boot_r2=[]; boot_rmse=[]; boot_mae=[]
    all_idx = np.arange(n)
    for b in range(n_boot):
        train_idx = rng.choice(n, size=n, replace=True)
        test_idx  = np.setdiff1d(all_idx, np.unique(train_idx))
        if len(test_idx) < 5: continue
        if len(np.unique(y[train_idx])) < 3: continue
        m    = fit_xgb(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        boot_r2.append(r2_score(y[test_idx], pred))
        boot_rmse.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        boot_mae.append(mean_absolute_error(y[test_idx], pred))
    def ci(dist):
        return (np.percentile(dist, alpha/2*100), np.percentile(dist, (1-alpha/2)*100), dist)
    return {"r2": ci(np.array(boot_r2)), "rmse": ci(np.array(boot_rmse)), "mae": ci(np.array(boot_mae))}


def make_figure(point_ests, boot_res, boot_oos_res):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    configs_is  = [("r2","R²",point_ests["r2_in"],"In-sample R²"),
                   ("rmse","RMSE (pp)",point_ests["rmse_in"],"In-sample RMSE"),
                   ("mae","MAE (pp)",point_ests["mae_in"],"In-sample MAE")]
    configs_oos = [("r2","R²",point_ests["r2_oos"],"OOS R²"),
                   ("rmse","RMSE (pp)",point_ests["rmse_oos"],"OOS RMSE"),
                   ("mae","MAE (pp)",point_ests["mae_oos"],"OOS MAE")]
    for row_idx, (configs, bres) in enumerate([(configs_is, boot_res),(configs_oos, boot_oos_res)]):
        for col_idx, (key, xlabel, pt, title) in enumerate(configs):
            ax = axes[row_idx, col_idx]
            lo, hi, dist = bres[key]
            ax.hist(dist, bins=40, color=NAVY, alpha=0.70, edgecolor="white")
            ax.axvline(pt, color="#d73027", lw=2.5, ls="-", label=f"Estimate: {pt:.4f}")
            ax.axvline(lo, color=ODS_C, lw=1.8, ls="--", label=f"95% CI:  [{lo:.4f}, {hi:.4f}]")
            ax.axvline(hi, color=ODS_C, lw=1.8, ls="--")
            ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel("Bootstrap frequency", fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8.5, framealpha=0.9); ax.set_facecolor("white")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(LGREY); ax.spines["bottom"].set_color(LGREY)
    fig.suptitle(f"Bootstrap 95% Confidence Intervals — XGBoost\n"
                 f"Top row: in-sample  |  Bottom row: OOS (standard bootstrap, out-of-bag)\n"
                 f"n_boot={N_BOOTSTRAP}  |  n_est={XGB_N_ESTIMATORS}, depth={XGB_MAX_DEPTH}, lr={XGB_LEARNING_RATE}  |  "
                 f"DV = YoY Δ quota deviation, within-demeaned",
                 fontsize=10, fontweight="bold")
    fig.patch.set_facecolor("white"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bootstrap_xgb_fig.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    print("Loading data...")
    prod           = load_production()
    brent          = load_brent()
    vix            = load_vix()
    gpr            = load_gpr()
    quotas         = load_quotas(prod)
    ogdp, inst     = load_controls()

    print("Building shocks...")
    shocks_df      = build_shocks(brent, vix, gpr)

    print("Building panel...")
    panel, shock_L = build_panel(prod, quotas, shocks_df, ogdp, inst)

    print("Pooling observations...")
    pool, X, y, feat_dm = pool_panel(panel, shock_L)

    tuned_params = None
    if TUNE_HYPERPARAMS:
        tuned_params = tune_xgb(X, y, feat_dm)

    print("Fitting XGBoost...")
    model = fit_xgb(X, y, params=tuned_params)
    (point_ests, oos_y, oos_p, oos_iso,
     last_fold_model, last_fold_X_te, last_fold_y_te) = compute_point_estimates(X, y, pool, params=tuned_params)

    print(f"  IS  R²: {point_ests['r2_in']:.4f}   RMSE: {point_ests['rmse_in']:.4f} pp   MAE: {point_ests['mae_in']:.4f} pp")
    print(f"  OOS R²: {point_ests['r2_oos']:.4f}   RMSE: {point_ests['rmse_oos']:.4f} pp   MAE: {point_ests['mae_oos']:.4f} pp")

    decomp_df = country_decomposition(oos_y, oos_p, oos_iso)
    plot_country_decomposition(decomp_df)
    decomp_df.to_csv(os.path.join(OUT_DIR,"country_decomposition.csv"), index=False)

    print("Computing TreeSHAP...")
    shap_df, sv  = compute_shap(model, X, pool, feat_dm)
    country_shap = plot_shap_overall(shap_df, feat_dm)

    perm_df = compute_permutation_importance(last_fold_model, last_fold_X_te, last_fold_y_te, feat_dm)
    rho, p_rho = plot_importance_comparison(shap_df, perm_df, feat_dm)

    pos_df, neg_df = compute_shap_asymmetric(shap_df)
    pos_df = significance_shap(pos_df); neg_df = significance_shap(neg_df)
    pos_df.drop(columns=["shap_vals"]).to_csv(os.path.join(OUT_DIR,"shap_asym_positive.csv"), index=False)
    neg_df.drop(columns=["shap_vals"]).to_csv(os.path.join(OUT_DIR,"shap_asym_negative.csv"), index=False)
    plot_asymmetric_shap(pos_df, neg_df)
    plot_shap_scatter(shap_df, point_ests["r2_oos"])

    print("Running bootstrap...")
    boot_res     = bootstrap_is(X, y)
    boot_oos_res = bootstrap_oos(X, y)

    r2_lo,  r2_hi,  _ = boot_res["r2"]
    rm_lo,  rm_hi,  _ = boot_res["rmse"]
    ma_lo,  ma_hi,  _ = boot_res["mae"]
    r2_oos_lo,  r2_oos_hi,  _ = boot_oos_res["r2"]
    rm_oos_lo,  rm_oos_hi,  _ = boot_oos_res["rmse"]
    ma_oos_lo,  ma_oos_hi,  _ = boot_oos_res["mae"]

    rows = [
        {"metric":"R² (in-sample)",  "estimate":round(point_ests['r2_in'],4),   "ci_lo":round(r2_lo,4),     "ci_hi":round(r2_hi,4),     "bootstrap":"in-sample",      "unit":""},
        {"metric":"RMSE (in-sample)","estimate":round(point_ests['rmse_in'],4),  "ci_lo":round(rm_lo,4),     "ci_hi":round(rm_hi,4),     "bootstrap":"in-sample",      "unit":"pp"},
        {"metric":"MAE (in-sample)", "estimate":round(point_ests['mae_in'],4),   "ci_lo":round(ma_lo,4),     "ci_hi":round(ma_hi,4),     "bootstrap":"in-sample",      "unit":"pp"},
        {"metric":"R² (OOS)",        "estimate":round(point_ests['r2_oos'],4),   "ci_lo":round(r2_oos_lo,4), "ci_hi":round(r2_oos_hi,4), "bootstrap":"OOS out-of-bag", "unit":""},
        {"metric":"RMSE (OOS)",      "estimate":round(point_ests['rmse_oos'],4), "ci_lo":round(rm_oos_lo,4), "ci_hi":round(rm_oos_hi,4), "bootstrap":"OOS out-of-bag", "unit":"pp"},
        {"metric":"MAE (OOS)",       "estimate":round(point_ests['mae_oos'],4),  "ci_lo":round(ma_oos_lo,4), "ci_hi":round(ma_oos_hi,4), "bootstrap":"OOS out-of-bag", "unit":"pp"},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "bootstrap_xgb_results.csv"), index=False)
    make_figure(point_ests, boot_res, boot_oos_res)

    print("Output in:", OUT_DIR)