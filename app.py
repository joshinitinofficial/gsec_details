import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Indian G-Sec Yield Analyzer",
    page_icon="📈",
    layout="wide"
)

st.write("✅ App started")  # important for Streamlit Cloud
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

try:
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
        m = re.match(r"(\d+)", symbol)
        if not m:
            return None
        raw = m.group(1)
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
    # YTM (SEMI-ANNUAL, NO SCIPY)
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
            )

        low, high = 0.0001, 0.20
        for _ in range(100):
            mid = (low + high) / 2
            if bond_price(mid) > price:
                low = mid
            else:
                high = mid

        return mid * 100

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
    spread = max_row["YTM_%"] - min_row["YTM_%"]

    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Max YTM Bond", max_row["SYMBOL"], f"{max_row['YTM_%']:.2f}%")
    c2.metric("🧊 Min YTM Bond", min_row["SYMBOL"], f"{min_row['YTM_%']:.2f}%")
    c3.metric("📐 YTM Spread", f"{spread:.2f}%", "Max − Min")

    st.markdown("---")

    # ==========================
    # TABLE (CENTERED & CLEAN)
    # ==========================
    st.subheader("📈 Top 10 Most Liquid Government Bonds")
    
    ui_df = final_df.rename(columns={
        "SYMBOL": "Bond",
        "COUPON_RATE": "Coupon (%)",
        "LTP": "LTP",
        "VOLUME": "Volume",
        "REDEMPTION_DATE_DISPLAY": "Redemption Date",
        "YEARS_TO_MATURITY": "Years to Maturity",
        "YTM_%": "YTM (%)"
    })
    
    # Format for display
    ui_df["Coupon (%)"] = ui_df["Coupon (%)"].map(lambda x: f"{x:.2f}")
    ui_df["LTP"] = ui_df["LTP"].map(lambda x: f"{x:.2f}")
    ui_df["YTM (%)"] = ui_df["YTM (%)"].map(lambda x: f"{x:.2f}")
    ui_df["Years to Maturity"] = ui_df["Years to Maturity"].map(lambda x: f"{x:.2f}")
    ui_df["Volume"] = ui_df["Volume"].map(lambda x: f"{int(x):,}")
    
    table_html = """
    <style>
    /* FORCE OVERRIDE STREAMLIT STYLES */
    table.gsec-table {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 8px !important;
        font-size: 15px !important;
    }
    
    table.gsec-table thead th {
        text-align: center !important;
        font-weight: 700 !important;
        padding: 12px !important;
        background: rgba(255,255,255,0.06) !important;
        border-bottom: 1px solid rgba(255,255,255,0.15) !important;
    }
    
    table.gsec-table tbody tr {
        background: rgba(255,255,255,0.02) !important;
    }
    
    table.gsec-table tbody td {
        text-align: center !important;
        padding: 12px !important;
        border-top: 1px solid rgba(255,255,255,0.08) !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    }
    
    table.gsec-table tbody tr:hover {
        background: rgba(255,255,255,0.08) !important;
    }
    </style>
    
    <table class="gsec-table">
    <thead>
    <tr>
    """
    
    for col in ui_df.columns:
        table_html += f"<th>{col}</th>"
    
    table_html += "</tr></thead><tbody>"
    
    for _, row in ui_df.iterrows():
        table_html += "<tr>"
        for val in row:
            table_html += f"<td>{val}</td>"
        table_html += "</tr>"
    
    table_html += "</tbody></table>"
    
    st.markdown(table_html, unsafe_allow_html=True)

    

    # ==========================
    # DOWNLOAD
    # ==========================
    st.download_button(
        "⬇️ Download Results",
        final_df.to_csv(index=False).encode("utf-8"),
        "gsec_ytm_output.csv",
        "text/csv"
    )

except Exception as e:
    st.error("❌ App crashed during execution")
    st.exception(e)

st.caption("Built for Indian Fixed Income Markets 🇮🇳")
