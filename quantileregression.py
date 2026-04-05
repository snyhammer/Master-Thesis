"""
OPEC Oil Production — Quantile Regression pooled
==========================================
DV:  ln(oil production, TBPD) — OPEC monthly panel
IVs: GPR shock (Δln GPR), Oil price shock (Δln Brent),
     VIX, Oil/GDP (%), Institutional Quality Index

Quantiles: 0.10, 0.25, 0.50, 0.75, 0.90
Country fixed effects via within-demeaning.

Requirements: pip install pandas numpy statsmodels matplotlib openpyxl xlrd
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# ── 0. File paths ─────────────────────────────────────────────────────────────
PROD_CSV   = "COil_production_monthly.csv"
BRENT_XLSX = "DCOILBRENTEU.xlsx"
GPR_XLS    = "GPR.xls"
VIX_XLSX   = "VIXCLS.xlsx"
OILGDP_XLS = "Oil_%_of_GDP.xls"
INST_XLSX  = "Institutional_Quality_Index.xlsx"

SAMPLE_START = "2000-01-01"
SAMPLE_END   = "2023-12-01"
QUANTILES    = [0.10, 0.25, 0.50, 0.75, 0.90]

OPEC = {
    "SAU": "Saudi Arabia", "IRN": "Iran",      "IRQ": "Iraq",
    "KWT": "Kuwait",       "ARE": "UAE",        "VEN": "Venezuela",
    "NGA": "Nigeria",      "DZA": "Algeria",   "LBY": "Libya",
    "GAB": "Gabon",        "COG": "Congo",     "GNQ": "Eq. Guinea",
    "AGO": "Angola",
}

# ── 1. Load & clean each dataset ──────────────────────────────────────────────

def load_production():
    """
    EIA CSV: columns come in pairs (even=date, odd=series key/values).
    Series keys are in row 1 of the ODD column. Data starts at row 9.
    """
    raw = pd.read_csv(PROD_CSV, sep=";", header=None, low_memory=False)
    records = []
    for pair in range(raw.shape[1] // 2):
        date_col = pair * 2
        val_col  = pair * 2 + 1
        key = str(raw.iloc[1, val_col])
        iso3 = next((c for c in OPEC if f"-{c}-" in key), None)
        if iso3 is None or "INTL.53-1-" not in key:
            continue
        dates  = raw.iloc[9:, date_col].astype(str)
        values = pd.to_numeric(raw.iloc[9:, val_col], errors="coerce")
        records.append(pd.DataFrame({"date": dates, "prod": values, "iso3": iso3}))

    df = pd.concat(records).dropna()
    df = df[df["date"].str.match(r"\d{4}-\d{2}")]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    return df

def load_brent():
    df = pd.read_excel(BRENT_XLSX, parse_dates=[0])
    df.columns = ["date", "brent"]
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.groupby("date")["brent"].mean().reset_index()
    df["d_ln_brent"] = np.log(df["brent"]).diff()
    return df

def load_gpr():
    df = pd.read_excel(GPR_XLS, engine="xlrd")[["month", "GPR"]]
    df.columns = ["date", "gpr"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna()
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["d_ln_gpr"] = np.log(df["gpr"]).diff()
    return df

def load_vix():
    df = pd.read_excel(VIX_XLSX, parse_dates=[0])
    df.columns = ["date", "vix"]
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.groupby("date")["vix"].mean().reset_index()

def load_oil_gdp():
    """
    World Bank format: header at row 3, columns = Country Name, Country Code,
    Indicator Name, Indicator Code, 1960, 1961, ..., 2023.
    Melt to long format on year.
    """
    df = pd.read_excel(OILGDP_XLS, engine="xlrd", header=3)
    df = df.rename(columns={"Country Code": "iso3"})
    year_cols = [c for c in df.columns if str(c).isdigit()
                 and 1990 <= int(c) <= 2030]
    df_long = df[["iso3"] + year_cols].melt(
        id_vars="iso3", var_name="year", value_name="oil_gdp"
    )
    df_long["year"] = df_long["year"].astype(int)
    df_long = df_long[df_long["iso3"].isin(OPEC)].dropna(subset=["oil_gdp"])
    return df_long

def load_inst():
    """Composite governance index averaged from WGI columns."""
    df = pd.read_excel(INST_XLSX)
    name_map = {
        "Algeria": "DZA", "Angola": "AGO", "Congo": "COG",
        "Equatorial Guinea": "GNQ", "Gabon": "GAB", "Iran": "IRN",
        "Iraq": "IRQ", "Kuwait": "KWT", "Libya": "LBY",
        "Nigeria": "NGA", "Saudi Arabia": "SAU",
        "United Arab Emirates": "ARE", "Venezuela": "VEN",
    }
    df["iso3"] = df["Country"].map(name_map)
    df = df.dropna(subset=["iso3"])
    wgi = [c for c in df.columns if any(k in c for k in
           ["effectiveness", "corruption", "Regulatory", "Political stability"])]
    df["inst"] = df[wgi].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df[["iso3", "year", "inst"]].dropna()

# ── 2. Build panel ────────────────────────────────────────────────────────────

prod    = load_production()
brent   = load_brent()
gpr     = load_gpr()
vix     = load_vix()
oil_gdp = load_oil_gdp()
inst    = load_inst()

panel = (prod
         .merge(brent,   on="date", how="left")
         .merge(gpr,     on="date", how="left")
         .merge(vix,     on="date", how="left"))

panel["year"] = panel["date"].dt.year
panel = (panel
         .merge(oil_gdp, on=["iso3", "year"], how="left")
         .merge(inst,    on=["iso3", "year"], how="left"))

panel = panel[(panel["date"] >= SAMPLE_START) & (panel["date"] <= SAMPLE_END)]
panel["ln_prod"] = np.log(panel["prod"].replace(0, np.nan))

KEY_VARS = ["ln_prod", "d_ln_gpr", "d_ln_brent", "vix", "oil_gdp", "inst"]
panel = panel.dropna(subset=KEY_VARS)

# Within-demean for country fixed effects
for v in KEY_VARS:
    panel[f"{v}_dm"] = panel[v] - panel.groupby("iso3")[v].transform("mean")

print(f"Panel: {len(panel)} obs | {panel['iso3'].nunique()} countries | "
      f"{panel['date'].min().date()} – {panel['date'].max().date()}")

# ── 3. Descriptive statistics ─────────────────────────────────────────────────

print("\n── Descriptive Statistics ──")
print(panel[KEY_VARS].describe().round(3))

# ── 4. Quantile regressions ───────────────────────────────────────────────────

FORMULA = ("ln_prod_dm ~ d_ln_gpr_dm + d_ln_brent_dm + "
           "vix_dm + oil_gdp_dm + inst_dm")

results = {}
for q in QUANTILES:
    results[q] = smf.quantreg(FORMULA, data=panel).fit(q=q, max_iter=2000)

print("\n── Quantile Regression Results ──")
print(f"{'':30}" + "".join(f"   τ={q:.2f}  " for q in QUANTILES))
print("-" * 82)

for var, label in [
    ("d_ln_gpr_dm",   "GPR Shock (Δln GPR)"),
    ("d_ln_brent_dm", "Oil Price Shock (Δln Brent)"),
    ("vix_dm",        "VIX"),
    ("oil_gdp_dm",    "Oil Rents (% GDP)"),
    ("inst_dm",       "Institutional Quality"),
]:
    row_b = f"{label:<30}"
    row_p = f"{'':30}"
    for q in QUANTILES:
        b   = results[q].params[var]
        p   = results[q].pvalues[var]
        sig = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
        row_b += f"  {b:+.4f}{sig:<3}"
        row_p += f"  ({p:.3f})   "
    print(row_b)
    print(row_p)

print("-" * 82)
print("*** p<0.01  ** p<0.05  * p<0.10")
print("Note: Country FE via within-demeaning; asymptotic standard errors.")

# ── 5. Plot coefficient profiles ──────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for ax, var, title in [
    (ax1, "d_ln_gpr_dm",   "GPR Shock (Δln GPR)"),
    (ax2, "d_ln_brent_dm", "Oil Price Shock (Δln Brent)"),
]:
    coefs = [results[q].params[var] for q in QUANTILES]
    lo    = [results[q].conf_int().loc[var, 0] for q in QUANTILES]
    hi    = [results[q].conf_int().loc[var, 1] for q in QUANTILES]

    ax.plot(QUANTILES, coefs, "o-", color="#2166ac", linewidth=2, markersize=7)
    ax.fill_between(QUANTILES, lo, hi, alpha=0.15, color="#2166ac")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Quantile (τ)")
    ax.set_ylabel("Coefficient")
    ax.set_xticks(QUANTILES)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Effect on ln(OPEC Oil Production) Across Quantiles",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_quantile_coefficients.png", dpi=150, bbox_inches="tight")
print("\nSaved: fig_quantile_coefficients.png")
plt.show()

# ── 6. Export results ─────────────────────────────────────────────────────────

rows = []
for q in QUANTILES:
    res = results[q]
    for var in res.params.index:
        rows.append({
            "quantile": q, "variable": var,
            "coef": res.params[var], "se": res.bse[var],
            "pvalue": res.pvalues[var],
            "ci_lo": res.conf_int().loc[var, 0],
            "ci_hi": res.conf_int().loc[var, 1],
        })

pd.DataFrame(rows).to_csv("results_quantile_regression.csv", index=False)
print("Saved: results_quantile_regression.csv")