import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO

sns.set_style("darkgrid")

st.set_page_config(
    page_title="InsightBolt - Fintech Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 InsightBolt - Fintech Data Analysis Dashboard")

# Sidebar - File Upload & Settings
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload your financial data (CSV, Excel)", type=["csv", "xlsx", "xls"]
)

# Sidebar - Data preview & filter options will come here
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.sidebar.error(f"Failed to load file: {e}")
        return None

def parse_dates(df):
    # Try to detect date columns intelligently
    date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
    if date_cols:
        df['Date'] = df[date_cols[0]]  # pick the first date column
    else:
        df['Date'] = pd.NaT
    return df

def get_amount_column(df):
    # Common variants
    candidates = ['Amount', 'Withdraw', 'Deposit', 'Credit', 'Debit', 'Balance']
    for c in candidates:
        if c in df.columns:
            return c
    # Try case insensitive search
    for col in df.columns:
        if any(word.lower() == col.lower() for word in candidates):
            return col
    return None

def main():
    df = load_data(uploaded_file)
    if df is None:
        st.info("Upload a file to get started.")
        return

    df = parse_dates(df)

    st.sidebar.markdown("### Data Preview & Filters")
    st.sidebar.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # Show sample of data with option to expand
    if st.sidebar.checkbox("Show raw data sample"):
        st.dataframe(df.head(100))

    # Numeric columns for filtering
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    date_col = 'Date' if 'Date' in df.columns and df['Date'].notna().any() else None

    # Filters
    if date_col:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        start_date, end_date = st.sidebar.date_input(
            "Filter by Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        if start_date > end_date:
            st.sidebar.error("Start date must be before end date")
            return
        df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

    if numeric_cols:
        st.sidebar.markdown("### Numeric Filters")
        for col in numeric_cols:
            min_val, max_val = float(df[col].min()), float(df[col].max())
            selected_range = st.sidebar.slider(
                f"{col} range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            df = df[(df[col] >= selected_range[0]) & (df[col] <= selected_range[1])]

    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

    # Show correlation matrix heatmap
    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Matrix")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    amount_col = get_amount_column(df)

    if amount_col:
        st.subheader(f"📈 {amount_col} Over Time")
        if date_col:
            # Interactive time series with Plotly
            fig = px.line(df, x=date_col, y=amount_col, title=f"{amount_col} Over Time", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid Date column to plot time series.")
    else:
        st.warning("No suitable Amount/Transaction column found for time series plot.")

    # Distribution plot
    if amount_col:
        st.subheader(f"📉 {amount_col} Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df[amount_col].dropna(), bins=50, kde=True, ax=ax)
        st.pyplot(fig)

    # Scatter plot - customizable by user
    if numeric_cols:
        st.subheader("🔎 Explore Relationships")
        x_axis = st.selectbox("X-axis", numeric_cols, index=0)
        y_axis = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        color_col = st.selectbox("Color by (optional)", [None] + df.columns.tolist())
        scatter_title = f"{y_axis} vs {x_axis}"
        if color_col:
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, title=scatter_title)
        else:
            fig = px.scatter(df, x=x_axis, y=y_axis, title=scatter_title)
        st.plotly_chart(fig, use_container_width=True)

    # Export cleaned/filtered data option
    st.sidebar.markdown("### Export Filtered Data")
    to_export = st.sidebar.button("Download CSV")
    if to_export:
        csv = df.to_csv(index=False).encode()
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="insightbolt_filtered.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
