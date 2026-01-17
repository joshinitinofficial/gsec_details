import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.optimize import newton
from pathlib import Path

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Indian G-Sec Yield Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("🇮🇳 Indian Government Bond Yield Analyzer")
st.caption("Exact maturity-based YTM | Semi-annual coupon | NSE data")
st.markdown("---")

# ==========================
# LOAD STATIC DEBT MASTER
# ==========================
DATA_PATH = Path("data/DEBT.csv")

if not DATA_PATH.exists():
    st.error("❌ DEBT.csv not found in /data folder")
    st.stop()

master = pd.read_csv(DATA_PATH)
master.columns = master.columns.str.strip()

master["REDEMPTION DATE"] = pd.to_datetime(
    master["REDEMPTION DATE"], errors="coerce"
)

master = master[["SYMBOL", "REDEMPTION DATE"]].dropna()

# ==========================
# USER UPLOAD
# ==========================
trade_file = st.file_uploader(
    "📥 Upload MW G-Sec Trading CSV",
    type=["csv"]
)

if not trade_file:
    st.info("⬆️ Please upload MW G-Sec CSV to continue")
    st.stop()

# ==========================
# LOAD & CLEAN TRADE DATA
# ==========================
df = pd.read_csv(trade_file)
df.columns = df.columns.str.strip()

df["LTP"] = pd.to_numeric(df["LTP"], errors="coerce")
df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce")

df = df.dropna(subset=["LTP", "VOLUME"])
df = df[df["SYMBOL"].str.contains("GS|GR", regex=True, na=False)]

# ==========================
# TOP 10 BY VOLUME
# ==========================
df = df.sort_values("VOLUME", ascending=False).head(10).copy()

# ==========================
# MERGE WITH MASTER
# ==========================
df = df.merge(master, on="SYMBOL", how="left")
df = df.dropna(subset=["REDEMPTION DATE"])

# ==========================
# COUPON PARSER
# ==========================
def parse_coupon(symbol):
    match = re.match(r"(\d+)", symbol)
    if not match:
        return None
    raw = match.group(1)
    return int(raw) / 10 if len(raw) == 2 else int(raw) / 100

df["COUPON_RATE"] = df["SYMBOL"].apply(parse_coupon)
df = df.dropna(subset=["COUPON_RATE"])

# ==========================
# YEARS TO MATURITY
# ==========================
today = pd.Timestamp.today().normalize()
df["YEARS_TO_MATURITY"] = (
    (df["REDEMPTION DATE"] - today).dt.days / 365
)

df = df[df["YEARS_TO_MATURITY"] > 0]

# ==========================
# YTM (SEMI-ANNUAL)
# ==========================
def calculate_ytm(price, coupon_rate, years, face_value=100):
    periods = int(round(years * 2))
    if price <= 0 or periods <= 0:
        return np.nan

    coupon = (coupon_rate / 2) * face_value

    def bond_price(y):
        return (
            sum(coupon / (1 + y / 2) ** t for t in range(1, periods + 1))
            + face_value / (1 + y / 2) ** periods
            - price
        )

    try:
        return newton(bond_price, 0.07, maxiter=100) * 100
    except:
        return np.nan

df["YTM_%"] = df.apply(
    lambda r: calculate_ytm(
        r["LTP"], r["COUPON_RATE"] / 100, r["YEARS_TO_MATURITY"]
    ),
    axis=1
)

# ==========================
# DATE FORMAT FOR UI
# ==========================
def format_date(d):
    if pd.isna(d):
        return ""
    day = d.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return d.strftime(f"%d{suffix} %b %Y")

df["REDEMPTION_DATE_DISPLAY"] = df["REDEMPTION DATE"].apply(format_date)

# ==========================
# FINAL DATAFRAME
# ==========================
final_df = df[
    [
        "SYMBOL",
        "COUPON_RATE",
        "LTP",
        "VOLUME",
        "REDEMPTION_DATE_DISPLAY",
        "YEARS_TO_MATURITY",
        "YTM_%"
    ]
].sort_values("YTM_%", ascending=False).reset_index(drop=True)

# ==========================
# METRICS
# ==========================
max_row = final_df.loc[final_df["YTM_%"].idxmax()]
min_row = final_df.loc[final_df["YTM_%"].idxmin()]
ytm_spread = max_row["YTM_%"] - min_row["YTM_%"]

c1, c2, c3 = st.columns(3)

c1.metric("🏆 Max YTM Bond", max_row["SYMBOL"], f"{max_row['YTM_%']:.2f}%")
c2.metric("🧊 Min YTM Bond", min_row["SYMBOL"], f"{min_row['YTM_%']:.2f}%")
c3.metric("📐 YTM Spread", f"{ytm_spread:.2f}%", "Max − Min")

st.markdown("---")

# ==========================
# TABLE (UI POLISHED)
# ==========================
st.subheader("📈 Top 10 Most Liquid Government Bonds")

display_df = final_df.rename(columns={
    "SYMBOL": "Bond",
    "COUPON_RATE": "Coupon (%)",
    "LTP": "LTP",
    "VOLUME": "Volume",
    "REDEMPTION_DATE_DISPLAY": "Redemption Date",
    "YEARS_TO_MATURITY": "Years to Maturity",
    "YTM_%": "YTM (%)"
})

styled_df = (
    display_df
    .style
    .format({
        "Coupon (%)": "{:.2f}",
        "LTP": "{:.2f}",
        "YTM (%)": "{:.2f}",
        "Years to Maturity": "{:.2f}",
        "Volume": "{:,.0f}"
    })
    .set_properties(**{"text-align": "center"})
    .set_table_styles([
        {"selector": "th", "props": [("text-align", "center"), ("font-weight", "600")]},
        {"selector": "td", "props": [("text-align", "center")]}
    ])
)

st.dataframe(styled_df, use_container_width=True, hide_index=True)

# ==========================
# DOWNLOAD
# ==========================
st.download_button(
    "⬇️ Download Results",
    final_df.to_csv(index=False).encode("utf-8"),
    "gsec_ytm_output.csv",
    "text/csv"
)

st.caption("Built for Indian Fixed Income Markets 🇮🇳")
