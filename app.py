# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('df_clean.csv')
    # Ensure 'Timestamp' is datetime if needed for future features, though not directly used in this app
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

df_clean = load_data()

# --- 2. Streamlit App Title ---
st.title('Process Parameter Distribution Dashboard')

# --- 3. Sidebar for Setpoint Selection ---
unique_setpoints = sorted(df_clean['Setpoint (barg)'].unique())
selected_setpoint = st.sidebar.selectbox('Select Setpoint (barg)', unique_setpoints)

# --- 4. Filter DataFrame based on selected setpoint ---
df_selected_setpoint = df_clean[df_clean['Setpoint (barg)'] == selected_setpoint].copy()

# --- 5. Function to Calculate Setpoint Statistics ---
def calculate_setpoint_stats(df_subset, sp_value):
    stats = {}

    if df_subset.empty or len(df_subset) < 2:
        return None # Indicate no sufficient data

    # a. Calculate means for P1, PID, Xt Non Choked
    stats['Inlet Pressure P1 (barg)_mean'] = df_subset['Inlet Pressure P1 (barg)'].mean()
    stats['PID Valve Output (%)_mean'] = df_subset['PID Valve Output (%)'].mean()
    stats['Xt Non Choked_mean'] = df_subset['Xt Non Choked'].mean()

    # b. Calculate mean and std for Outlet Pressure P2
    stats['Outlet Pressure P2_mean'] = df_subset['Outlet Pressure P2'].mean()
    stats['Outlet Pressure P2_std'] = df_subset['Outlet Pressure P2'].std()

    # c. Calculate USL and LSL
    stats['USL'] = sp_value + 0.5
    stats['LSL'] = sp_value - 0.5

    # d. Calculate UCL and LCL, handle zero std
    if stats['Outlet Pressure P2_std'] == 0 or np.isnan(stats['Outlet Pressure P2_std']):
        stats['UCL'] = np.nan
        stats['LCL'] = np.nan
    else:
        stats['UCL'] = stats['Outlet Pressure P2_mean'] + 3 * stats['Outlet Pressure P2_std']
        stats['LCL'] = stats['Outlet Pressure P2_mean'] - 3 * stats['Outlet Pressure P2_std']

    # e. Calculate accuracy_from_control_limits
    if not np.isnan(stats['UCL']) and not np.isnan(stats['LCL']):
        stats['Accuracy (UCL/LCL)'] = (stats['UCL'] - stats['LCL']) / 2
    else:
        stats['Accuracy (UCL/LCL)'] = np.nan

    return stats

# --- 6. Function to Plot Distributions ---
def plot_distributions(df_subset, selected_sp_value, stats):
    if stats is None:
        st.warning(f"No sufficient data to calculate statistics and plot for Setpoint {selected_sp_value} barg.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Distribution Plots for Setpoint: {selected_sp_value} Barg', fontsize=20, y=1.02)

    column_features = [
        'Inlet Pressure P1 (barg)',
        'PID Valve Output (%)',
        'Xt Non Choked',
        'Outlet Pressure P2'
    ]

    colors = {
        'Inlet Pressure P1 (barg)': 'skyblue',
        'PID Valve Output (%)': 'lightgreen',
        'Xt Non Choked': 'lightcoral',
        'Outlet Pressure P2': 'orange'
    }

    for i, feature in enumerate(column_features):
        ax = axes.flatten()[i]
        sns.histplot(df_subset[feature], bins=30, kde=True, stat='density', color=colors[feature], edgecolor='black', alpha=0.7, ax=ax)

        # Add mean line
        if feature == 'Inlet Pressure P1 (barg)':
            mean_val = stats['Inlet Pressure P1 (barg)_mean']
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.set_xlabel('Inlet Pressure P1 (barg)')
            ax.set_ylabel('Density')

        elif feature == 'PID Valve Output (%)':
            mean_val = stats['PID Valve Output (%)_mean']
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.set_xlabel('PID Valve Output (%)')
            ax.set_ylabel('Density')

        elif feature == 'Xt Non Choked':
            mean_val = stats['Xt Non Choked_mean']
            ax.axvline(mean_val, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.set_xlabel('Xt Non Choked')
            ax.set_ylabel('Density')

        elif feature == 'Outlet Pressure P2':
            mean_val = stats['Outlet Pressure P2_mean']
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')

            # Add USL/LSL and UCL/LCL lines for Outlet Pressure P2
            ax.axvline(stats['USL'], color='green', linestyle='dashed', linewidth=2, label=f'USL: {stats["USL"]:.2f}')
            ax.axvline(stats['LSL'], color='green', linestyle='dashed', linewidth=2, label=f'LSL: {stats["LSL"]:.2f}')

            if not np.isnan(stats['UCL']) and not np.isnan(stats['LCL']):
                ax.axvline(stats['UCL'], color='purple', linestyle='dashed', linewidth=2, label=f'UCL: {stats["UCL"]:.2f}')
                ax.axvline(stats['LCL'], color='purple', linestyle='dashed', linewidth=2, label=f'LCL: {stats["LCL"]:.2f}')
                ax.text(0.05, 0.95, f'+- Acc (UCL/LCL): {stats["Accuracy (UCL/LCL)"]:.2f} barg',
                        transform=ax.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            ax.set_xlabel('Outlet Pressure P2 (barg)')
            ax.set_ylabel('Density')
        
        ax.set_title(f'{feature}')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

# --- 7. Main Application Logic ---
if not df_selected_setpoint.empty:
    st.subheader(f"Displaying Data for Setpoint: {selected_setpoint} barg")
    
    # Calculate stats for the selected setpoint
    setpoint_stats = calculate_setpoint_stats(df_selected_setpoint, selected_setpoint)

    # Plot distributions
    plot_distributions(df_selected_setpoint, selected_setpoint, setpoint_stats)
else:
    st.warning(f"No data available for Setpoint: {selected_setpoint} barg. Please select another setpoint.")

