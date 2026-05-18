import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import shap

np.random.seed(42)

SVR_C       = 100
SVR_EPSILON = 0.01
SVR_GAMMA   = 0.001
SVR_KERNEL  = "rbf"

LAG_SHOCKS  = 9
CUM_WINDOW  = 6
YOY_HORIZON = 12

N_BOOTSTRAP  = 1000
BLOCK_LENGTH = 12
CI_ALPHA     = 0.05

MIN_OBS      = 30
ROLL_WINDOW  = 616

SAMPLE_START = "1993-01-01"
SAMPLE_END   = "2025-12-01"

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(DATA_DIR, "bootstrap_pooled_output")
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

NAVY="#1B2A6B"; STEEL="#4A6FA5"; GREEN="#1a9850"; LGREY="#cccccc"
ODS_C="#e07b39"; ORS_C="#9467bd"; GPR_C="#2ca02c"

COUNTRY_COLOURS = {
    "Algeria":"#1f77b4","Iraq":"#ff7f0e","Kuwait":"#2ca02c",
    "UAE":"#d62728","Venezuela":"#9467bd","Nigeria":"#8c564b",
    "Libya":"#e377c2","Gabon":"#7f7f7f","Congo":"#bcbd22",
    "Eq. Guinea":"#17becf","Saudi Arabia":"#aec7e8","Iran":"#ffbb78",
}

FEAT_LABELS = {
    "ODS_L":           "Oil demand shock",
    "ORS_L":           "Oil risk shock",
    "GPR_L":           "Geopolitical risk",
    "oil_gdp":         "Oil share of GDP",
    "inst":            "Institutional quality",
    "d_pol_stab":      "Change in political stability",
    "quota_tightness": "Quota tightness",
    "yoy_dev_l1":      "Lagged YoY deviation",
}


def load_production():
    raw = pd.read_csv(PATH_PRODUCTION, sep=None, engine="python", header=None)
    opec_key = {iso: f"INTL.53-1-{iso}-TBPD.M" for iso in OPEC}
    col_map = {}
    for ci in range(raw.shape[1]):
        for ri in range(5):
            for iso, key in opec_key.items():
                if key in str(raw.iloc[ri, ci]): col_map[iso] = ci
    records = []
    for iso, kc in col_map.items():
        dc = kc - 1; ds = None
        for r in range(raw.shape[0]):
            v = str(raw.iloc[r, dc])
            if len(v) == 7 and v[4] == "-" and v[:4].isdigit(): ds = r; break
        if ds is None: continue
        dates = pd.to_datetime(raw.iloc[ds:, dc].astype(str), format="%Y-%m", errors="coerce")
        vals  = pd.to_numeric(raw.iloc[ds:, kc], errors="coerce")
        records.append(pd.DataFrame({"date":dates,"prod":vals,"iso3":iso}).dropna(subset=["date","prod"]))
    prod = pd.concat(records).sort_values(["iso3","date"]).reset_index(drop=True)
    prod["date"] = prod["date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    return prod

def load_quotas(prod):
    import re
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
            s,e=pm(parts[0]),pm(parts[1])
            if s and e: return pd.date_range(s,e,freq="MS").tolist()
        elif len(parts)==1:
            t=pm(parts[0]); return [t] if t else []
        return []
    qr=[]
    for ri in range(5,18):
        cn=str(df_q.iloc[ri,0]).strip(); iso3=ciso.get(cn)
        if not iso3: continue
        for ci,hdr in enumerate(ch,start=1):
            val=str(df_q.iloc[ri,ci]).strip()
            if val in ["nd","–","-","nan","NaN",""]: continue
            try: qv=float(val.replace(",",""))
            except: continue
            for ts in pp(hdr): qr.append({"iso3":iso3,"date":ts,"quota":qv})
    quotas_raw=pd.DataFrame(qr)
    prod_med=prod.groupby("iso3")["prod"].median()
    clean=[r for _,r in quotas_raw.iterrows()
           if r["quota"]>=0.15*prod_med.get(r["iso3"],np.nan)]
    return pd.DataFrame(clean).sort_values(["iso3","date"]).reset_index(drop=True)

def load_brent():
    df=pd.read_excel(PATH_BRENT); df.columns=["date","brent"]
    df["date"]=pd.to_datetime(df["date"])
    return df.dropna().set_index("date")["brent"].resample("MS").mean().reset_index()

def load_vix():
    df=pd.read_excel(PATH_VIX); df.columns=["date","vix"]
    df["date"]=pd.to_datetime(df["date"])
    return df.dropna().set_index("date")["vix"].resample("MS").mean().reset_index()

def load_gpr():
    df=pd.read_excel(PATH_GPR,engine="xlrd")
    df["date"]=pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
    return df[["date","GPR"]].dropna().sort_values("date").reset_index(drop=True)

def load_controls():
    df_og=pd.read_excel(PATH_OIL_GDP,engine="xlrd",skiprows=3)
    yc=[c for c in df_og.columns if str(c).strip().isdigit()]
    ogdp=df_og.melt(id_vars=["Country Name","Country Code"],value_vars=yc,
                    var_name="year",value_name="oil_gdp")
    ogdp["year"]=ogdp["year"].astype(int)
    ogdp=ogdp.rename(columns={"Country Code":"iso3"})[["iso3","year","oil_gdp"]].dropna()
    df_i=pd.read_excel(PATH_INST)
    pc="Political stability index (-2.5 weak; 2.5 strong)"
    gc="Government effectiveness index (-2.5 weak; 2.5 strong)"
    df_i=df_i.rename(columns={"Code":"iso3","Year":"year",pc:"pol_stab",gc:"gov_eff"})
    df_i["inst"]=df_i[["pol_stab","gov_eff"]].mean(axis=1)
    return ogdp, df_i[["iso3","year","pol_stab","inst"]].dropna(subset=["inst"])


def ar1_residual(series):
    s=series.dropna().reset_index(drop=True)
    lag=s.shift(1); mask=lag.notna()
    coef=np.polyfit(lag[mask],s[mask],1)
    resid=s-(coef[0]*lag+coef[1])
    return (resid-resid.mean())/(resid.std()+1e-9)

def build_shocks(brent,vix,gpr):
    df=brent.merge(vix,on="date",how="inner").sort_values("date").reset_index(drop=True)
    df["d_ln_brent"]=np.log(df["brent"]).diff(); df=df.dropna().reset_index(drop=True)
    try: arma=ARIMA(df["vix"].values,order=(1,0,1)).fit(); ors=arma.resid
    except: ors=df["vix"].diff().fillna(0).values
    ors=(ors-ors.mean())/(ors.std()+1e-9); df["ORS"]=ors
    X=np.column_stack([np.ones(len(df)),ors])
    c,*_=np.linalg.lstsq(X,df["d_ln_brent"].values,rcond=None)
    ods=df["d_ln_brent"].values-X@c
    df["ODS"]=(ods-ods.mean())/ods.std()
    g=gpr.copy(); g["ln_gpr"]=np.log(g["GPR"].clip(lower=0.01))
    g["GPR"]=ar1_residual(g["ln_gpr"]).values
    df=df.merge(g[["date","GPR"]],on="date",how="left")
    return df[["date","ODS","ORS","GPR"]]


def build_panel(prod,quotas,shocks,ogdp,inst):
    base=prod.merge(quotas,on=["iso3","date"],how="inner")
    base["dev_pct"]=(base["prod"]-base["quota"])/base["quota"]*100
    base["quota_tightness"]=base["quota"]/(
        base.groupby("iso3")["prod"].transform(
            lambda x:x.rolling(12,min_periods=3).mean())+1e-9)
    base=base.sort_values(["iso3","date"])
    base["yoy_dev"]=base.groupby("iso3")["dev_pct"].transform(lambda x:x-x.shift(YOY_HORIZON))
    base["yoy_dev_l1"]=base.groupby("iso3")["yoy_dev"].transform(lambda x:x.shift(1))
    p=base.merge(shocks,on="date",how="left")
    shock_L=[]
    for col in SHOCKS:
        out=col+"_L"
        for iso3 in OPEC:
            mask=p["iso3"]==iso3
            p.loc[mask,out]=(p.loc[mask,col].rolling(CUM_WINDOW,min_periods=1).mean().shift(LAG_SHOCKS).values)
        shock_L.append(out)
    p["year"]=p["date"].dt.year
    p=p.merge(ogdp[["iso3","year","oil_gdp"]],on=["iso3","year"],how="left")
    p=p.merge(inst[["iso3","year","pol_stab","inst"]],on=["iso3","year"],how="left")
    p=p.sort_values(["iso3","date"])
    p["d_pol_stab"]=p.groupby("iso3")["pol_stab"].transform(lambda x:x.diff())
    p=p[(p["date"]>=SAMPLE_START)&(p["date"]<=SAMPLE_END)].copy()
    for col in ["oil_gdp","inst","d_pol_stab","quota_tightness"]:
        p[col]=p.groupby("iso3")[col].transform(lambda x:x.ffill().bfill())
    return p.reset_index(drop=True), shock_L


def pool_panel(panel, shock_L):
    rows=[]
    for iso3 in OPEC:
        sub=panel[panel["iso3"]==iso3].dropna(
            subset=["yoy_dev","yoy_dev_l1"]+shock_L+CTRL_COLS).copy()
        if len(sub)<MIN_OBS: continue
        rows.append(sub)
    pool=pd.concat(rows).sort_values("date").reset_index(drop=True)
    feat_cols=shock_L+CTRL_COLS+["yoy_dev_l1"]
    for col in feat_cols:
        pool[f"{col}_dm"] = pool[col] - pool.groupby("iso3")[col].transform("mean")
    pool["yoy_dev_dm"] = pool["yoy_dev"] - pool.groupby("iso3")["yoy_dev"].transform("mean")
    dm_cols=[f"{c}_dm" for c in feat_cols]
    X_raw=pool[dm_cols].values
    y=pool["yoy_dev_dm"].values
    return pool, X_raw, y, dm_cols


def compute_point_estimates(X_raw, y, pool):
    sc = StandardScaler(); X = sc.fit_transform(X_raw)
    svr = SVR(C=SVR_C, epsilon=SVR_EPSILON, gamma=SVR_GAMMA, kernel=SVR_KERNEL).fit(X, y)
    pred_in = svr.predict(X)
    r2_in   = r2_score(y, pred_in)
    rmse_in = np.sqrt(mean_squared_error(y, pred_in))
    mae_in  = mean_absolute_error(y, pred_in)
    tscv = TimeSeriesSplit(n_splits=5, max_train_size=ROLL_WINDOW)
    oos_y = []; oos_p = []; oos_iso = []
    for tr, te in tscv.split(X_raw):
        sc2 = StandardScaler()
        svr2 = SVR(C=SVR_C, epsilon=SVR_EPSILON, gamma=SVR_GAMMA, kernel=SVR_KERNEL).fit(
            sc2.fit_transform(X_raw[tr]), y[tr])
        oos_y.extend(y[te])
        oos_p.extend(svr2.predict(sc2.transform(X_raw[te])))
        oos_iso.extend(pool["iso3"].values[te])
    oos_y   = np.array(oos_y); oos_p = np.array(oos_p); oos_iso = np.array(oos_iso)
    r2_oos   = r2_score(oos_y, oos_p)
    rmse_oos = np.sqrt(mean_squared_error(oos_y, oos_p))
    mae_oos  = mean_absolute_error(oos_y, oos_p)
    return ({"r2_in":r2_in,"rmse_in":rmse_in,"mae_in":mae_in,
             "r2_oos":r2_oos,"rmse_oos":rmse_oos,"mae_oos":mae_oos},
            oos_y, oos_p, oos_iso)


def block_bootstrap(X_raw, y, oos_y=None, oos_p=None, n_boot=N_BOOTSTRAP, block_len=BLOCK_LENGTH, alpha=CI_ALPHA):
    n=len(y); n_blocks=int(np.ceil(n/block_len))
    boot_r2=[]; boot_rmse=[]; boot_mae=[]
    boot_r2_oos=[]; boot_rmse_oos=[]; boot_mae_oos=[]
    do_oos = (oos_y is not None and oos_p is not None and len(oos_y)>0)
    n_oos  = len(oos_y) if do_oos else 0
    nb_oos = int(np.ceil(n_oos/block_len)) if do_oos else 0
    rng = np.random.default_rng(42)
    for b in range(n_boot):
        starts=rng.integers(0,max(1,n-block_len+1),size=n_blocks)
        idx=np.concatenate([np.arange(s,min(s+block_len,n)) for s in starts])[:n]
        Xb=X_raw[idx]; yb=y[idx]
        if len(np.unique(yb))<3: continue
        sc=StandardScaler(); Xb_s=sc.fit_transform(Xb)
        svr=SVR(C=SVR_C,epsilon=SVR_EPSILON,gamma=SVR_GAMMA,kernel=SVR_KERNEL).fit(Xb_s,yb)
        pred=svr.predict(Xb_s)
        boot_r2.append(r2_score(yb,pred))
        boot_rmse.append(np.sqrt(mean_squared_error(yb,pred)))
        boot_mae.append(mean_absolute_error(yb,pred))
        if do_oos:
            starts_o=rng.integers(0,max(1,n_oos-block_len+1),size=nb_oos)
            idx_o=np.concatenate([np.arange(s,min(s+block_len,n_oos)) for s in starts_o])[:n_oos]
            yb_o=oos_y[idx_o]; pb_o=oos_p[idx_o]
            if len(np.unique(yb_o))>=3:
                boot_r2_oos.append(r2_score(yb_o,pb_o))
                boot_rmse_oos.append(np.sqrt(mean_squared_error(yb_o,pb_o)))
                boot_mae_oos.append(mean_absolute_error(yb_o,pb_o))
        if (b+1)%100==0: print(f"    {b+1}/{n_boot} done", flush=True)
    def ci(dist):
        dist=np.array(dist)
        return (np.percentile(dist,alpha/2*100), np.percentile(dist,(1-alpha/2)*100), dist)
    result = {"r2":ci(boot_r2),"rmse":ci(boot_rmse),"mae":ci(boot_mae)}
    if do_oos and len(boot_r2_oos)>10:
        result["r2_oos"]   = ci(boot_r2_oos)
        result["rmse_oos"] = ci(boot_rmse_oos)
        result["mae_oos"]  = ci(boot_mae_oos)
    return result


def compute_perm_importance(X_raw, y, dm_cols, n_repeats=50):
    sc          = StandardScaler(); X_sc = sc.fit_transform(X_raw)
    svr         = SVR(C=SVR_C, epsilon=SVR_EPSILON, gamma=SVR_GAMMA, kernel=SVR_KERNEL).fit(X_sc, y)
    baseline_r2 = r2_score(y, svr.predict(X_sc))
    rng         = np.random.default_rng(42)
    n_features  = X_sc.shape[1]
    importances = np.zeros((n_features, n_repeats))
    for fi in range(n_features):
        for rep in range(n_repeats):
            X_perm        = X_sc.copy()
            X_perm[:, fi] = rng.permutation(X_perm[:, fi])
            importances[fi, rep] = baseline_r2 - r2_score(y, svr.predict(X_perm))
    feat_names = [c.replace("_dm","") for c in dm_cols]
    perm_df = pd.DataFrame({
        "feature":    feat_names,
        "label":      [FEAT_LABELS.get(f, f) for f in feat_names],
        "importance": importances.mean(axis=1),
        "std":        importances.std(axis=1),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    perm_df.to_csv(os.path.join(OUT_DIR, "perm_importance_svr.csv"), index=False)
    for suffix, df_plot in [("full", perm_df), ("zoom", perm_df[perm_df["feature"]!="yoy_dev_l1"].copy())]:
        df_p = df_plot.sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(4, len(df_p)*0.65)))
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")
        ax.barh(df_p["label"], df_p["importance"], xerr=df_p["std"], color=NAVY, alpha=0.85,
                edgecolor="white", capsize=3, error_kw=dict(elinewidth=1, ecolor="grey", capthick=1))
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Mean R\u00b2 decrease when feature is permuted", fontsize=10)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.2); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"perm_importance_svr_{suffix}.png"), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
    return perm_df


def country_error_analysis(oos_y, oos_p, oos_iso):
    rows = []
    for iso3, name in OPEC.items():
        mask = oos_iso == iso3
        if mask.sum() < 5: continue
        yt = oos_y[mask]; yp = oos_p[mask]
        rows.append({"iso3":iso3,"country":name,
                     "r2_oos":  round(float(r2_score(yt,yp)),4),
                     "rmse_oos":round(float(np.sqrt(mean_squared_error(yt,yp))),4),
                     "mae_oos": round(float(mean_absolute_error(yt,yp)),4),
                     "n":       int(mask.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "country_errors_svr.csv"), index=False)
    df_rmse = df.sort_values("rmse_oos", ascending=False)
    df_mae  = df.sort_values("mae_oos",  ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df_plot, metric, ylabel in [
        (axes[0], df_rmse, "rmse_oos", "RMSE (pp)"),
        (axes[1], df_mae,  "mae_oos",  "MAE (pp)"),
    ]:
        x    = np.arange(len(df_plot))
        cols = [COUNTRY_COLOURS.get(n, STEEL) for n in df_plot["country"]]
        ax.bar(x, df_plot[metric], color=cols, alpha=0.88, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(df_plot["country"], rotation=40, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Country-level OOS {ylabel} — SVR\n(walk-forward, 5-fold TimeSeriesSplit)", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.18); ax.set_facecolor("white")
    fig.patch.set_facecolor("white"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "country_errors_svr.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return df


def plot_pred_vs_actual(oos_y, oos_p, oos_iso):
    import matplotlib as mpl; mpl.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for iso3, name in OPEC.items():
        mask = oos_iso == iso3
        if mask.sum() < 3: continue
        ax.scatter(oos_y[mask], oos_p[mask], color=COUNTRY_COLOURS.get(name, STEEL),
                   alpha=0.55, s=22, edgecolors="none", zorder=3)
    all_vals = np.concatenate([oos_y, oos_p])
    lim_lo = np.percentile(all_vals, 1); lim_hi = np.percentile(all_vals, 99)
    pad = (lim_hi - lim_lo) * 0.05; lim = (lim_lo-pad, lim_hi+pad)
    ax.plot(lim, lim, color="black", lw=1.2, ls="--", zorder=2)
    ax.axhline(0, color=LGREY, lw=0.7, zorder=1); ax.axvline(0, color=LGREY, lw=0.7, zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed YoY Δ deviation (pp)", fontsize=11)
    ax.set_ylabel("Predicted YoY Δ deviation (pp)", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LGREY); ax.spines["bottom"].set_color(LGREY)
    ax.grid(alpha=0.15, zorder=0); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pred_vs_actual_svr.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_residuals(oos_y, oos_p, oos_iso):
    import matplotlib as mpl; mpl.rcParams["axes.unicode_minus"] = False
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    country_list = [(iso3, name) for iso3, name in OPEC.items() if (oos_iso==iso3).sum()>=3]
    for xi, (iso3, name) in enumerate(country_list):
        mask   = oos_iso == iso3
        resids = oos_p[mask] - oos_y[mask]
        jitter = rng.uniform(-0.25, 0.25, size=mask.sum())
        ax.scatter(xi+jitter, resids, color=COUNTRY_COLOURS.get(name, STEEL), alpha=0.55, s=18, edgecolors="none", zorder=3)
    ax.axhline(0, color="black", lw=1.0, ls="--", zorder=2)
    ax.set_xlim(-0.6, len(country_list)-0.4); ax.set_xticks([])
    ax.set_xlabel("Country (pooled OOS observations)", fontsize=11)
    ax.set_ylabel("Residual (predicted - actual, pp)", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LGREY); ax.spines["bottom"].set_color(LGREY)
    ax.grid(axis="y", alpha=0.15, zorder=0); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "residuals_svr.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def compute_shap(X_raw, y, pool, feat_cols, n_background=50, nsamples=200):
    sc  = StandardScaler(); X_sc = sc.fit_transform(X_raw)
    svr = SVR(C=SVR_C, epsilon=SVR_EPSILON, gamma=SVR_GAMMA, kernel=SVR_KERNEL).fit(X_sc, y)
    rng    = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_sc), size=n_background, replace=False)
    background  = shap.kmeans(X_sc[bg_idx], min(10, n_background))
    explainer   = shap.KernelExplainer(svr.predict, background)
    shap_vals   = explainer.shap_values(X_sc, nsamples=nsamples, l1_reg="num_features(5)")
    shap_df = pool[["iso3","date","yoy_dev"]].copy().reset_index(drop=True)
    shap_df["country"] = shap_df["iso3"].map(OPEC)
    for fi, col in enumerate(feat_cols):
        label = col.replace("_dm","")
        shap_df[f"shap_{label}"] = shap_vals[:, fi]
        shap_df[f"raw_{label}"]  = X_raw[:, fi]
    return shap_df

def significance_shap(vals, n_boot=N_BOOTSTRAP, block_len=BLOCK_LENGTH):
    vals = np.array(vals, dtype=float); vals = vals[np.isfinite(vals)]
    n    = len(vals)
    if n < 3: return 1.0, ""
    rng      = np.random.default_rng(42)
    n_blocks = int(np.ceil(n / block_len))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, n-block_len+1), size=n_blocks)
        idx    = np.concatenate([np.arange(s, min(s+block_len, n)) for s in starts])[:n]
        boot_means[b] = vals[idx].mean()
    prop_pos = (boot_means > 0).mean(); prop_neg = (boot_means < 0).mean()
    p        = float(min(2*min(prop_pos, prop_neg), 1.0))
    stars    = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
    return p, stars

def plot_asymmetric_shap(shap_df, feat_cols):
    shock_feats  = [c for c in feat_cols if any(s in c for s in ["ODS","ORS","GPR"])]
    shock_labels = {"ODS_L":"ODS (Demand shock)","ORS_L":"ORS (Risk shock)","GPR_L":"GPR (Geopolitical risk)"}
    countries    = list(COUNTRY_COLOURS.keys())
    for sf in shock_feats:
        raw_col  = f"raw_{sf.replace('_dm','')}"
        shap_col = f"shap_{sf.replace('_dm','')}"
        label    = sf.replace("_dm","")
        for sign, sign_label, fname_suffix in [
            ("pos", "Positive Shock Months (shock > 0)", "positive"),
            ("neg", "Negative Shock Months (shock <= 0)", "negative"),
        ]:
            mask  = shap_df[raw_col] > 0 if sign == "pos" else shap_df[raw_col] <= 0
            sub   = shap_df[mask].copy()
            rows  = []
            for iso3, name in OPEC.items():
                g = sub[sub["iso3"]==iso3][shap_col].dropna()
                if len(g) < 3: continue
                p, stars = significance_shap(g.values)
                rows.append({"country":name,"mean_shap":g.mean(),"p_value":p,"stars":stars})
            if not rows: continue
            rdf = pd.DataFrame(rows)
            rdf = rdf.set_index("country").reindex([c for c in countries if c in rdf.index])
            rdf = rdf.dropna(subset=["mean_shap"])
            fig, ax = plt.subplots(figsize=(12, 5))
            x    = np.arange(len(rdf))
            cols = [COUNTRY_COLOURS.get(c,"#888") for c in rdf.index]
            ax.bar(x, rdf["mean_shap"].values, color=cols, alpha=0.88, edgecolor="white", linewidth=0.5)
            yspan = rdf["mean_shap"].abs().max()*0.07 if len(rdf) else 0.1
            for xi, (_, row) in zip(x, rdf.iterrows()):
                if row.get("stars",""):
                    ypos = row["mean_shap"]+(yspan if row["mean_shap"]>=0 else -yspan)
                    ax.text(xi, ypos, row["stars"], ha="center",
                            va="bottom" if row["mean_shap"]>=0 else "top", fontsize=10, fontweight="bold")
            ax.axhline(0, color="black", lw=0.9, ls="--", zorder=0)
            ax.set_title(f"{shock_labels.get(label,label)} — {sign_label}\nMean KernelSHAP contribution to YoY Δ deviation  |  SVR",
                         fontsize=12, fontweight="bold")
            ax.set_ylabel("Mean SHAP contribution (pp)", fontsize=10)
            ax.set_xticks(x); ax.set_xticklabels(rdf.index, rotation=45, ha="right", fontsize=9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.18); ax.set_facecolor("white")
            handles = [plt.Rectangle((0,0),1,1,color=COUNTRY_COLOURS.get(c,"#888"),alpha=0.88) for c in rdf.index]
            ax.legend(handles, list(rdf.index), loc="center left", bbox_to_anchor=(1.01,0.5),
                      fontsize=9, title="Country", title_fontsize=9, framealpha=0.95, edgecolor=LGREY)
            fig.patch.set_facecolor("white"); plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"shap_asym_{label}_{fname_suffix}.png"),
                        dpi=200, bbox_inches="tight", facecolor="white")
            plt.close()
    shock_feats_raw  = [f"raw_{sf.replace('_dm','')}"  for sf in shock_feats]
    shock_feats_shap = [f"shap_{sf.replace('_dm','')}" for sf in shock_feats]
    for sign, suffix in [("pos","positive"),("neg","negative")]:
        rows_out = []
        for sf in shock_feats:
            raw_col  = f"raw_{sf.replace('_dm','')}"
            shap_col = f"shap_{sf.replace('_dm','')}"
            label    = sf.replace("_dm","")
            mask = shap_df[raw_col]>0 if sign=="pos" else shap_df[raw_col]<=0
            sub  = shap_df[mask]
            for iso3, name in OPEC.items():
                g = sub[sub["iso3"]==iso3][shap_col].dropna()
                if len(g)<3: continue
                p, stars = significance_shap(g.values)
                rows_out.append({"country":name,"iso3":iso3,"shock":label,
                                 "mean_shap":round(g.mean(),4),"mean_abs_shap":round(g.abs().mean(),4),
                                 "n":len(g),"p_value":round(p,6),"stars":stars})
        pd.DataFrame(rows_out).to_csv(os.path.join(OUT_DIR,f"shap_asym_{suffix}.csv"), index=False)


def plot_shap_over_time(shap_df):
    import matplotlib as mpl; mpl.rcParams["axes.unicode_minus"] = False
    shock_configs = [
        ("shap_ODS_L", "ODS SHAP contribution (pp)", "shap_time_ods.png"),
        ("shap_ORS_L", "ORS SHAP contribution (pp)", "shap_time_ors.png"),
        ("shap_GPR_L", "GPR SHAP contribution (pp)", "shap_time_gpr.png"),
    ]
    for shap_col, ylabel, fname in shock_configs:
        if shap_col not in shap_df.columns: continue
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")
        ax.axhline(0, color="black", lw=0.8, ls="--", zorder=1)
        for iso3, name in OPEC.items():
            sub = (shap_df[shap_df["iso3"]==iso3][["date",shap_col]].dropna().sort_values("date"))
            if len(sub) < 6: continue
            smoothed = sub[shap_col].rolling(window=6, min_periods=3).mean()
            ax.plot(sub["date"], smoothed, color=COUNTRY_COLOURS.get(name, STEEL), lw=1.4, alpha=0.85, zorder=2)
        ax.set_xlabel("Year", fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(LGREY); ax.spines["bottom"].set_color(LGREY)
        ax.grid(axis="y", alpha=0.12, zorder=0); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()


def make_figure(point_ests, boot_res):
    has_oos = "r2_oos" in boot_res
    ncols   = 6 if has_oos else 3
    fig, axes = plt.subplots(1, ncols, figsize=(ncols*4.5, 5))
    configs = [
        ("r2",   "R²",        point_ests["r2_in"],  "In-sample R²"),
        ("rmse", "RMSE (pp)", point_ests["rmse_in"], "In-sample RMSE"),
        ("mae",  "MAE (pp)",  point_ests["mae_in"],  "In-sample MAE"),
    ]
    if has_oos:
        configs += [
            ("r2_oos",   "R²",        point_ests["r2_oos"],  "OOS R²"),
            ("rmse_oos", "RMSE (pp)", point_ests["rmse_oos"], "OOS RMSE"),
            ("mae_oos",  "MAE (pp)",  point_ests["mae_oos"],  "OOS MAE"),
        ]
    for ax, (key, xlabel, pt, title) in zip(axes, configs):
        lo, hi, dist = boot_res[key]
        ax.hist(dist, bins=40, color=NAVY, alpha=0.70, edgecolor="white")
        ax.axvline(pt, color="#d73027", lw=2.5, ls="-", label=f"Estimate: {pt:.4f}")
        ax.axvline(lo, color=ODS_C, lw=1.8, ls="--", label=f"95% CI:  [{lo:.4f}, {hi:.4f}]")
        ax.axvline(hi, color=ODS_C, lw=1.8, ls="--")
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel("Bootstrap frequency", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5, framealpha=0.9); ax.set_facecolor("white")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(LGREY); ax.spines["bottom"].set_color(LGREY)
    fig.suptitle(f"Block Bootstrap 95% Confidence Intervals — SVR\n"
                 f"(n_boot={N_BOOTSTRAP}, block_len={BLOCK_LENGTH}m  |  C={SVR_C}, ε={SVR_EPSILON}, γ={SVR_GAMMA}  |  YoY Δ Deviation DV)",
                 fontsize=11, fontweight="bold")
    fig.patch.set_facecolor("white"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"bootstrap_pooled_fig.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    print("Loading data...")
    prod       = load_production()
    brent      = load_brent()
    vix        = load_vix()
    gpr        = load_gpr()
    quotas     = load_quotas(prod)
    ogdp, inst = load_controls()

    print("Building shocks...")
    shocks_df = build_shocks(brent, vix, gpr)

    print("Building panel...")
    panel, shock_L = build_panel(prod, quotas, shocks_df, ogdp, inst)

    print("Pooling observations...")
    pool, X_raw, y, feat_cols = pool_panel(panel, shock_L)

    print("Computing point estimates...")
    point_ests, oos_y, oos_p, oos_iso = compute_point_estimates(X_raw, y, pool)
    print(f"  IS  R²: {point_ests['r2_in']:.4f}   RMSE: {point_ests['rmse_in']:.4f} pp   MAE: {point_ests['mae_in']:.4f} pp")
    print(f"  OOS R²: {point_ests['r2_oos']:.4f}   RMSE: {point_ests['rmse_oos']:.4f} pp   MAE: {point_ests['mae_oos']:.4f} pp")

    country_err_df = country_error_analysis(oos_y, oos_p, oos_iso)
    plot_pred_vs_actual(oos_y, oos_p, oos_iso)
    plot_residuals(oos_y, oos_p, oos_iso)

    print("Running block bootstrap...")
    boot_res = block_bootstrap(X_raw, y, oos_y=oos_y, oos_p=oos_p)

    r2_lo,  r2_hi,  _ = boot_res["r2"]
    rm_lo,  rm_hi,  _ = boot_res["rmse"]
    ma_lo,  ma_hi,  _ = boot_res["mae"]
    r2_oos_lo=r2_oos_hi=rm_oos_lo=rm_oos_hi=ma_oos_lo=ma_oos_hi=None
    if "r2_oos" in boot_res:
        r2_oos_lo, r2_oos_hi, _ = boot_res["r2_oos"]
        rm_oos_lo, rm_oos_hi, _ = boot_res["rmse_oos"]
        ma_oos_lo, ma_oos_hi, _ = boot_res["mae_oos"]

    rows=[
        {"metric":"R² (in-sample)",  "estimate":round(point_ests['r2_in'],4),   "ci_lo":round(r2_lo,4),   "ci_hi":round(r2_hi,4),   "unit":""},
        {"metric":"RMSE (in-sample)","estimate":round(point_ests['rmse_in'],4),  "ci_lo":round(rm_lo,4),   "ci_hi":round(rm_hi,4),   "unit":"pp"},
        {"metric":"MAE (in-sample)", "estimate":round(point_ests['mae_in'],4),   "ci_lo":round(ma_lo,4),   "ci_hi":round(ma_hi,4),   "unit":"pp"},
        {"metric":"R² (OOS)",        "estimate":round(point_ests['r2_oos'],4),   "ci_lo":round(r2_oos_lo,4) if r2_oos_lo else "—", "ci_hi":round(r2_oos_hi,4) if r2_oos_hi else "—", "unit":""},
        {"metric":"RMSE (OOS)",      "estimate":round(point_ests['rmse_oos'],4), "ci_lo":round(rm_oos_lo,4) if rm_oos_lo else "—", "ci_hi":round(rm_oos_hi,4) if rm_oos_hi else "—", "unit":"pp"},
        {"metric":"MAE (OOS)",       "estimate":round(point_ests['mae_oos'],4),  "ci_lo":round(ma_oos_lo,4) if ma_oos_lo else "—", "ci_hi":round(ma_oos_hi,4) if ma_oos_hi else "—", "unit":"pp"},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"bootstrap_pooled_results.csv"), index=False)

    print("Computing permutation importance...")
    perm_df = compute_perm_importance(X_raw, y, feat_cols)

    print("Computing KernelSHAP...")
    shap_df = compute_shap(X_raw, y, pool, feat_cols)
    plot_asymmetric_shap(shap_df, feat_cols)
    plot_shap_over_time(shap_df)
    make_figure(point_ests, boot_res)

    print("Output in:", OUT_DIR)