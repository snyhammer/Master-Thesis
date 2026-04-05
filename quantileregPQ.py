#### per country

"""
OPEC Oil Production — Quantile Regression per Country
======================================================
Runs separate quantile regressions for each OPEC country.
DV:  ln(oil production, TBPD)
IVs: GPR shock (Δln GPR), Oil price shock (Δln Brent), VIX, Oil/GDP (%), Inst. Quality

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

# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_production():
    raw = pd.read_csv(PROD_CSV, sep=";", header=None, low_memory=False)
    records = []
    for pair in range(raw.shape[1] // 2):
        date_col, val_col = pair * 2, pair * 2 + 1
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
    df = pd.read_excel(OILGDP_XLS, engine="xlrd", header=3)
    df = df.rename(columns={"Country Code": "iso3"})
    year_cols = [c for c in df.columns if str(c).isdigit() and 1990 <= int(c) <= 2030]
    df_long = df[["iso3"] + year_cols].melt(id_vars="iso3", var_name="year", value_name="oil_gdp")
    df_long["year"] = df_long["year"].astype(int)
    return df_long[df_long["iso3"].isin(OPEC)].dropna(subset=["oil_gdp"])

def load_inst():
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

panel = (load_production()
         .merge(load_brent(), on="date", how="left")
         .merge(load_gpr(),   on="date", how="left")
         .merge(load_vix(),   on="date", how="left"))

panel["year"] = panel["date"].dt.year
panel = (panel
         .merge(load_oil_gdp(), on=["iso3", "year"], how="left")
         .merge(load_inst(),    on=["iso3", "year"], how="left"))

panel = panel[(panel["date"] >= SAMPLE_START) & (panel["date"] <= SAMPLE_END)]
panel["ln_prod"] = np.log(panel["prod"].replace(0, np.nan))

KEY_VARS = ["ln_prod", "d_ln_gpr", "d_ln_brent", "vix", "oil_gdp", "inst"]
panel = panel.dropna(subset=KEY_VARS)

print(f"Panel: {len(panel)} obs | {panel['iso3'].nunique()} countries | "
      f"{panel['date'].min().date()} – {panel['date'].max().date()}")

# ── 3. Per-country quantile regressions ───────────────────────────────────────

FORMULA = "ln_prod ~ d_ln_gpr + d_ln_brent + vix + oil_gdp + inst"
results  = {}   # {iso3: {q: QuantRegResults}}
skipped  = []

for iso3, name in OPEC.items():
    sub = panel[panel["iso3"] == iso3].copy()
    if len(sub) < 30:
        print(f"Skipping {name} ({iso3}): only {len(sub)} obs")
        skipped.append(iso3)
        continue
    results[iso3] = {}
    for q in QUANTILES:
        try:
            results[iso3][q] = smf.quantreg(FORMULA, data=sub).fit(q=q, max_iter=2000)
        except Exception as e:
            print(f"  {name} τ={q}: {e}")

# ── 4. Print results per country ──────────────────────────────────────────────

for iso3, res_dict in results.items():
    name = OPEC[iso3]
    n    = len(panel[panel["iso3"] == iso3])
    print(f"\n{'='*70}")
    print(f"  {name} ({iso3})  —  n={n}")
    print(f"{'='*70}")
    print(f"{'':30}" + "".join(f"   τ={q:.2f}  " for q in QUANTILES))
    print("-" * 70)
    for var, label in [
        ("d_ln_gpr",   "GPR Shock (Δln GPR)"),
        ("d_ln_brent", "Oil Price Shock (Δln Brent)"),
        ("vix",        "VIX"),
        ("oil_gdp",    "Oil Rents (% GDP)"),
        ("inst",       "Institutional Quality"),
    ]:
        row_b = f"{label:<30}"
        row_p = f"{'':30}"
        for q in QUANTILES:
            if q not in res_dict:
                row_b += "     N/A    "; row_p += "            "; continue
            b   = res_dict[q].params[var]
            p   = res_dict[q].pvalues[var]
            sig = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
            row_b += f"  {b:+.4f}{sig:<3}"
            row_p += f"  ({p:.3f})   "
        print(row_b)
        print(row_p)
    print("-" * 70)

print("\n*** p<0.01  ** p<0.05  * p<0.10")

# ── 5. Plot: GPR & Price shock coefficients across quantiles, by country ──────

focus = {
    "d_ln_gpr":   "GPR Shock",
    "d_ln_brent": "Oil Price Shock",
}

for var, var_label in focus.items():
    countries = list(results.keys())
    n_countries = len(countries)
    ncols = 3
    nrows = int(np.ceil(n_countries / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5),
                             sharey=False)
    axes = axes.flatten()

    for i, iso3 in enumerate(countries):
        ax   = axes[i]
        name = OPEC[iso3]
        res_dict = results[iso3]

        coefs = [res_dict[q].params[var] for q in QUANTILES if q in res_dict]
        lo    = [res_dict[q].conf_int().loc[var, 0] for q in QUANTILES if q in res_dict]
        hi    = [res_dict[q].conf_int().loc[var, 1] for q in QUANTILES if q in res_dict]
        qs    = [q for q in QUANTILES if q in res_dict]

        ax.plot(qs, coefs, "o-", color="#2166ac", linewidth=2, markersize=6)
        ax.fill_between(qs, lo, hi, alpha=0.15, color="#2166ac")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xticks(QUANTILES)
        ax.set_xticklabels([str(q) for q in QUANTILES], fontsize=8)
        ax.set_xlabel("τ", fontsize=9)
        ax.set_ylabel("Coefficient", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for j in range(n_countries, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{var_label} → ln(Oil Production) by Country",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"fig_{var}_by_country.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fname}")
    plt.show()

# ── 6. Export results ─────────────────────────────────────────────────────────

rows = []
for iso3, res_dict in results.items():
    for q, res in res_dict.items():
        for var in res.params.index:
            rows.append({
                "country": OPEC[iso3], "iso3": iso3, "quantile": q,
                "variable": var,
                "coef":   res.params[var],
                "se":     res.bse[var],
                "pvalue": res.pvalues[var],
                "ci_lo":  res.conf_int().loc[var, 0],
                "ci_hi":  res.conf_int().loc[var, 1],
            })

pd.DataFrame(rows).to_csv("results_per_country_quantile.csv", index=False)
print("Saved: results_per_country_quantile.csv")