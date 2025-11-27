import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Page setup
st.set_page_config(page_title="Airtrol Test Dashboard", layout="wide")

# ================= CONFIGURATION ================= #

DATA_DIR = "airtrol_tests"

# Hard-coded test configuration (as provided by user)
TEST_CONFIG = {
    # inlet 6.0
    13: (5.5, 6.0), 14: (5.5, 6.0), 15: (5.5, 6.0), 16: (5.5, 6.0), 17: (5.5, 6.0),
    18: (5.0, 6.0), 19: (5.0, 6.0), 20: (5.0, 6.0), 21: (5.0, 6.0),
    25: (4.5, 6.0), 26: (4.5, 6.0), 27: (4.5, 6.0),
    28: (4.0, 6.0), 29: (4.0, 6.0),

    # inlet 5.5
    30: (5.0, 5.5), 31: (5.0, 5.5), 32: (5.0, 5.5), 33: (5.0, 5.5), 34: (5.0, 5.5),
    35: (4.5, 5.5), 36: (4.5, 5.5), 37: (4.5, 5.5), 38: (4.5, 5.5),
    39: (4.0, 5.5),
    40: (5.0, 5.5), 41: (5.0, 5.5),
    42: (4.5, 5.5),
    43: (4.0, 5.5)
}

TEST_NUMBERS = sorted(TEST_CONFIG.keys())

# Find CSV files automatically
test_files = {
    int(f.split("_")[-1].split(".")[0]): os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.startswith("airtrol_test_") and f.endswith(".csv")
}

test_files = {k: v for k, v in test_files.items() if k in TEST_NUMBERS}


# ================= FUNCTIONS ================= #

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def hist_with_gaussian_and_mean(ax, series, title, xlabel, bins=30):
    data = series.dropna().values
    if len(data) == 0:
        ax.text(0.5,0.5,"No data",ha="center",va="center")
        return

    mean = data.mean()
    std = data.std(ddof=1)
    counts, edges, _ = ax.hist(data, bins=bins, alpha=0.6)

    x_vals = np.linspace(data.min(), data.max(), 300)
    bw = edges[1] - edges[0]
    pdf = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_vals-mean)/std)**2)
    ax.plot(x_vals, pdf * len(data) * bw)

    ax.axvline(mean, linestyle="--", label=f"Mean={mean:.3f}")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")


# ================= STREAMLIT LAYOUT ================= #

st.title("Airtrol Performance Dashboard")

with st.sidebar:
    st.header("Controls")
    selected = st.selectbox("Select Test Number", TEST_NUMBERS)
    bins = st.slider("Histogram bins", 10, 60, 30)

file_path = test_files[selected]
setpoint, inlet_pressure = TEST_CONFIG[selected]
USL = setpoint + 0.5
LSL = setpoint - 0.5

df = pd.read_csv(file_path)
df = ensure_numeric(df, [
    "Inlet Pressure P1 (barg)", "VFM Flow Rate (SCFM)",
    "Xt", "Outlet Pressure P2"
])

p1 = df["Inlet Pressure P1 (barg)"]
p2 = df["Outlet Pressure P2"]
vfm = df["VFM Flow Rate (SCFM)"]
xt = df["Xt"]

p2_mean = p2.mean()
p2_std = p2.std(ddof=1)
UCL = p2_mean + 3*p2_std
LCL = p2_mean - 3*p2_std
accuracy = np.mean(np.abs(p2 - setpoint))
within_spec = ((p2>=LSL)&(p2<=USL)).mean()*100


# ================= KPI SECTION ================= #

st.subheader(f"Test {selected} Summary")
col1,col2,col3,col4 = st.columns(4)
col1.metric("Inlet Pressure (barg)", f"{inlet_pressure:.2f}")
col2.metric("Setpoint (barg)", f"{setpoint:.2f}")
col3.metric("Outlet P2 Mean", f"{p2_mean:.3f}")
col4.metric("Accuracy ±barg", f"{accuracy:.3f}")

col5,col6,col7,col8 = st.columns(4)
col5.metric("Samples", len(df))
col6.metric("Std Dev P2", f"{p2_std:.4f}")
col7.metric("% Within Spec", f"{within_spec:.1f}%")
col8.metric("USL / LSL", f"{USL:.1f} / {LSL:.1f}")


# ================= 2x2 DASHBOARD ================= #

fig, ax = plt.subplots(2, 2, figsize=(14,10))

hist_with_gaussian_and_mean(ax[0,0], p1, "Inlet Pressure P1", "P1", bins)
hist_with_gaussian_and_mean(ax[0,1], vfm, "VFM Flow (SCFM)", "Flow", bins)
hist_with_gaussian_and_mean(ax[1,0], xt, "Xt Value", "Xt", bins)

# Outlet pressure with UCL/LCL/USL/LSL
ax2 = ax[1,1]
ax2.hist(p2.dropna(), bins=bins, alpha=0.6)

# Control + Spec limits
ax2.axvline(p2_mean, linestyle="-", label=f"Mean={p2_mean:.3f}")
ax2.axvline(UCL, linestyle="--", label=f"UCL={UCL:.3f}")
ax2.axvline(LCL, linestyle="--", label=f"LCL={LCL:.3f}")
ax2.axvline(USL, linestyle=":", label=f"USL={USL:.3f}")
ax2.axvline(LSL, linestyle=":", label=f"LSL={LSL:.3f}")

ax2.set_title("Outlet Pressure P2 (barg)")
ax2.set_xlabel("P2")
ax2.set_ylabel("Frequency")
ax2.legend()

plt.tight_layout()
st.pyplot(fig)

with st.expander("Show Raw Data"):
    st.dataframe(df.head())

