
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="Airtrol Test Dashboard",
    layout="wide"
)

# ---------------- Helper functions ----------------

def load_test_files(data_dir, test_numbers):
    file_map = {}
    for fname in os.listdir(data_dir):
        if fname.startswith("airtrol_test_") and fname.endswith(".csv"):
            try:
                num = int(fname.split("_")[-1].split(".")[0])
            except ValueError:
                continue
            if num in test_numbers:
                file_map[num] = os.path.join(data_dir, fname)
    return file_map


def ensure_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def hist_with_gaussian_and_mean(ax, series, title, xlabel, bins=30):
    data = series.dropna().values
    ax.set_title(title)

    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        return

    mean = data.mean()
    std = data.std(ddof=1)

    counts, bin_edges, _ = ax.hist(data, bins=bins, alpha=0.6)

    x_vals = np.linspace(data.min(), data.max(), 300)
    bin_width = bin_edges[1] - bin_edges[0]
    pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mean) / std) ** 2)
    pdf_scaled = pdf * len(data) * bin_width
    ax.plot(x_vals, pdf_scaled)

    ax.axvline(mean, linestyle="--")
    ax.text(
        mean,
        max(counts) * 0.9,
        f"Mean = {mean:.4f}",
        rotation=90,
        va="top",
        ha="right",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")


def compute_kpis(df):
    p1 = df["Inlet Pressure P1 (barg)"]
    p2 = df["Outlet Pressure P2"]
    vfm = df["VFM Flow Rate (SCFM)"]
    xt = df["Xt"]
    setpoint = df["Setpoint (barg)"].mode().iloc[0]

    USL = setpoint + 0.5
    LSL = setpoint - 0.5

    mean_p1 = p1.mean()
    std_p1 = p1.std(ddof=1)
    mean_p2 = p2.mean()
    std_p2 = p2.std(ddof=1)

    accuracy = np.mean(np.abs(p2 - setpoint))
    within_spec = ((p2 >= LSL) & (p2 <= USL)).mean() * 100.0

    return {
        "setpoint": setpoint,
        "USL": USL,
        "LSL": LSL,
        "mean_p1": mean_p1,
        "std_p1": std_p1,
        "mean_p2": mean_p2,
        "std_p2": std_p2,
        "accuracy": accuracy,
        "within_spec": within_spec,
        "mean_vfm": vfm.mean(),
        "mean_xt": xt.mean(),
        "n_samples": len(df),
    }


# ---------------- App configuration ----------------

DATA_DIR = "airtrol_tests"
TEST_NUMBERS = [
    31, 42, 33, 14, 40, 28, 15, 39, 17, 29,
    37, 27, 13, 35, 38, 16, 43, 18, 36, 21,
    32, 26, 30, 34, 20, 25, 19
]

test_file_map = load_test_files(DATA_DIR, TEST_NUMBERS)

# ---------------- Layout: UI ----------------

st.title("Airtrol Test Dashboards")

st.markdown(
    "**Select a test from the sidebar to view its quality performance and control behavior.**"
)

if not test_file_map:
    st.error("⚠️ No test files found in airtrol_tests/")
    st.stop()

with st.sidebar:
    st.header("Controls")
    tests = sorted(test_file_map.keys())
    selected_test = st.selectbox("Select Test Number", tests)
    bins = st.slider("Histogram bins", 10, 60, 30, 5)

file_path = test_file_map[selected_test]
df = pd.read_csv(file_path)

df = ensure_numeric(df, [
    "Inlet Pressure P1 (barg)", "VFM Flow Rate (SCFM)",
    "Xt", "Outlet Pressure P2", "Setpoint (barg)"
])

p1 = df["Inlet Pressure P1 (barg)"]
vfm = df["VFM Flow Rate (SCFM)"]
xt = df["Xt"]
p2 = df["Outlet Pressure P2"]

kpis = compute_kpis(df)

# ---------------- KPIs ----------------
st.subheader(f"Test {selected_test} Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", kpis["n_samples"])
col2.metric("Setpoint (barg)", f"{kpis['setpoint']:.3f}")
col3.metric("Outlet P2 Mean (barg)", f"{kpis['mean_p2']:.3f}")
col4.metric("Accuracy (±barg)", f"{kpis['accuracy']:.3f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("P1 Mean (barg)", f"{kpis['mean_p1']:.3f}")
col6.metric("P1 STD", f"{kpis['std_p1']:.4f}")
col7.metric("P2 STD", f"{kpis['std_p2']:.4f}")
col8.metric("Within Spec (%)", f"{kpis['within_spec']:.1f}%")

# ---------------- Dashboard ----------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

hist_with_gaussian_and_mean(axes[0, 0], p1, "Inlet Pressure P1 (barg)", "P1", bins)
hist_with_gaussian_and_mean(axes[0, 1], vfm, "VFM Flow Rate (SCFM)", "Flow", bins)
hist_with_gaussian_and_mean(axes[1, 0], xt, "Xt", "Xt Value", bins)

ax = axes[1, 1]
p2_data = p2.dropna().values

if len(p2_data) > 0:
    mean_p2 = p2_data.mean()
    std_p2 = p2_data.std(ddof=1)
    UCL = mean_p2 + 3 * std_p2
    LCL = mean_p2 - 3 * std_p2

    counts, edges, _ = ax.hist(p2_data, bins=bins, alpha=0.6)
    width = edges[1] - edges[0]
    x_vals = np.linspace(p2_data.min(), p2_data.max(), 300)
    pdf = (1/(std_p2*np.sqrt(2*np.pi)))*np.exp(-0.5*((x_vals-mean_p2)/std_p2)**2)
    pdf_scaled = pdf * len(p2_data) * width
    ax.plot(x_vals, pdf_scaled)

    ax.axvline(mean_p2, linestyle="-", label=f"Mean={mean_p2:.3f}")
    ax.axvline(UCL, "--", label=f"UCL={UCL:.3f}")
    ax.axvline(LCL, "--", label=f"LCL={LCL:.3f}")
    ax.axvline(kpis["USL"], ":", label=f"USL={kpis['USL']:.3f}")
    ax.axvline(kpis["LSL"], ":", label=f"LSL={kpis['LSL']:.3f}")

    ax.set_title("Outlet Pressure P2 (barg)")
    ax.set_xlabel("P2 Value")
    ax.set_ylabel("Frequency")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
st.pyplot(fig)

with st.expander("Show Data Preview"):
    st.dataframe(df.head())
