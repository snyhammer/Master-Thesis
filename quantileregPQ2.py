#### OPEC Oil Production Quantile Regression — FINAL ROBUST VERSION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 0. File paths (Matches your uploaded CSV exports) ────────────────────────
PROD_CSV   = "COil_production_monthly.csv"
BRENT_CSV  = "DCOILBRENTEU.xlsx"
GPR_CSV    = "GPR.xls"
VIX_CSV    = "VIXCLS.xlsx"
OILGDP_CSV = "Oil_%_of_GDP.xls"
INST_CSV   = "Institutional_Quality_Index.xlsx"

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
    # Production file uses ";" and has metadata at top
    raw = pd.read_csv(PROD_CSV, sep=";", header=None, low_memory=False, encoding='latin1')
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
    # Force 2 columns to avoid ParserError from trailing commas
    df = pd.read_csv(BRENT_CSV, encoding='latin1', on_bad_lines='skip', usecols=[0, 1])
    df.columns = ["date", "brent"]
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["brent"] = pd.to_numeric(df["brent"], errors='coerce')
    df = df.groupby("date")["brent"].mean().reset_index()
    df["d_ln_brent"] = np.log(df["brent"]).diff()
    return df

def load_gpr():
    df = pd.read_csv(GPR_CSV, encoding='latin1', on_bad_lines='skip')
    # Based on file inspection, 'month' contains numeric/date values
    df = df[["month", "GPR"]]
    df.columns = ["date", "gpr"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna()
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["d_ln_gpr"] = np.log(df["gpr"]).diff()
    return df

def load_vix():
    df = pd.read_csv(VIX_CSV, encoding='latin1', on_bad_lines='skip', usecols=[0, 1])
    df.columns = ["date", "vix"]
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df.dropna(subset=["date"])
    df["vix"] = pd.to_numeric(df["vix"], errors='coerce')
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.groupby("date")["vix"].mean().reset_index()
    df["ln_vix"] = np.log(df["vix"])
    return df

def load_oil_gdp():
    # World Bank file has 3 metadata rows
    df = pd.read_csv(OILGDP_CSV, encoding='latin1', skiprows=3)
    iso3_map = {
        "Saudi Arabia": "SAU", "Iran, Islamic Rep.": "IRN", "Iraq": "IRQ",
        "Kuwait": "KWT", "United Arab Emirates": "ARE", "Venezuela": "VEN",
        "Nigeria": "NGA", "Algeria": "DZA", "Libya": "LBY",
        "Gabon": "GAB", "Congo, Rep.": "COG", "Equatorial Guinea": "GNQ",
        "Angola": "AGO"
    }
    # Reshape year columns
    year_cols = [str(y) for y in range(1960, 2025)]
    df_long = df.melt(id_vars=["Country Name"], value_vars=[c for c in year_cols if c in df.columns], 
                      var_name="year", value_name="oil_gdp")
    df_long["iso3"] = df_long["Country Name"].map(iso3_map)
    df_long = df_long.dropna(subset=["iso3", "oil_gdp"])
    df_long["year"] = df_long["year"].astype(int)
    return df_long[["iso3", "year", "oil_gdp"]]

def load_inst():
    df = pd.read_csv(INST_CSV, encoding='latin1')
    name_map = {
        "Algeria": "DZA", "Angola": "AGO", "Congo": "COG",
        "Equatorial Guinea": "GNQ", "Gabon": "GAB", "Iran": "IRN",
        "Iraq": "IRQ", "Kuwait": "KWT", "Libya": "LBY",
        "Nigeria": "NGA", "Saudi Arabia": "SAU",
        "United Arab Emirates": "ARE", "Venezuela": "VEN",
    }
    df["iso3"] = df["Country"].map(name_map)
    df = df.dropna(subset=["iso3"])
    # Average WGI indicators
    wgi = [c for c in df.columns if any(k in c.lower() for k in
           ["effectiveness", "corruption", "regulatory", "stability"])]
    df["inst"] = df[wgi].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df[["iso3", "year", "inst"]].dropna()

# ── 2. Build panel ────────────────────────────────────────────────────────────

print("Starting data merge...")
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

# Lag and Standardize for clearer effects
panel = panel.sort_values(["iso3", "date"])
cols_to_lag = ["d_ln_gpr", "d_ln_brent", "ln_vix", "oil_gdp", "inst"]
for col in cols_to_lag:
    panel[f"L1_{col}"] = panel.groupby("iso3")[col].shift(1)

lagged_cols = [f"L1_{col}" for col in cols_to_lag]
panel = panel.dropna(subset=lagged_cols + ["ln_prod"])

scaler = StandardScaler()
panel[lagged_cols] = scaler.fit_transform(panel[lagged_cols])

print(f"Panel ready: {len(panel)} observations.")

# ── 3. Run and Plot ──────────────────────────────────────────────────────────

FORMULA = "ln_prod ~ L1_d_ln_gpr + L1_d_ln_brent + L1_ln_vix + L1_oil_gdp + L1_inst"
results = {}

for iso3, name in OPEC.items():
    sub = panel[panel["iso3"] == iso3]
    if len(sub) < 20: continue
    results[iso3] = {q: smf.quantreg(FORMULA, data=sub).fit(q=q) for q in QUANTILES}
    print(f"Processed: {name}")

# Plotting one key variable as an example
var = "L1_d_ln_gpr"
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()
for i, (iso3, res_dict) in enumerate(results.items()):
    if i >= len(axes): break
    qs = list(res_dict.keys())
    coefs = [res_dict[q].params[var] for q in qs]
    axes[i].plot(qs, coefs, 'o-', color='red')
    axes[i].axhline(0, color='black', ls='--')
    axes[i].set_title(OPEC[iso3])

plt.tight_layout()
plt.savefig("fig_GPR_shocks.png")
print("Saved visualization: fig_GPR_shocks.png")