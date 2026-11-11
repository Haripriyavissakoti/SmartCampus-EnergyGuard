"""
SmartCampus EnergyGuard - Streamlit Dashboard
=============================================
Interactive Web Application for Campus Electricity Usage Monitoring & ML Anomaly Detection.
DISCLAIMER: Simulated prototype data — not actual college electricity measurements.
Never claim that the system is connected to actual college electricity meters.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import report_generator

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="SmartCampus EnergyGuard | ML Energy Monitoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic & Viva-Ready Presentation
st.markdown("""
<style>
    /* Global Container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        padding: 22px 28px;
        border-radius: 14px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }
    
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-fill-color: transparent;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    
    .disclaimer-banner {
        display: flex;
        align-items: center;
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 2px 0;
    }
    .metric-sub {
        font-size: 0.80rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Alert Notification Banner */
    .alert-panel {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 6px solid #ef4444;
        padding: 14px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Academic Info Box */
    .academic-note {
        background: rgba(56, 189, 248, 0.08);
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.86rem;
        color: #cbd5e1;
        margin: 12px 0;
    }
    
    /* Inspector Card */
    .inspector-box {
        background: #1e293b;
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 20px;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA AND MODEL LOADING
# ==============================================================================
@st.cache_data
def load_dataset():
    """
    Loads enriched energy dataset. If missing, automatically runs data generation
    and model training pipelines.
    """
    csv_file = "data/smartcampus_energy_with_anomalies.csv"
    legacy_file = "detected_anomalies.csv"
    
    target_file = csv_file if os.path.exists(csv_file) else legacy_file
    
    if not os.path.exists(target_file):
        with st.spinner("Dataset not found. Generating simulated dataset and training model..."):
            import generate_data
            import train_model
            generate_data.generate_college_electricity_data()
            train_model.train_and_detect_anomalies()
            target_file = csv_file if os.path.exists(csv_file) else legacy_file
            
    df = pd.read_csv(target_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data
def load_evaluation_metrics():
    """Loads model evaluation report metrics."""
    metrics_path = "models/evaluation_metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

df_raw = load_dataset()
metrics_data = load_evaluation_metrics()

# Session State for Alert Acknowledgment
if "acknowledged_alerts" not in st.session_state:
    st.session_state.acknowledged_alerts = set()

# ==============================================================================
# 3. HEADER & SIMULATED DATA DISCLAIMER
# ==============================================================================
st.markdown("""
<div class="header-card">
    <div class="header-title">⚡ SMARTCAMPUS ENERGYGUARD</div>
    <div class="header-subtitle">ML-Based Campus Electricity Consumption Monitoring & Anomaly Detection System</div>
    <div class="disclaimer-banner">
        ⚠️ <b>Simulated prototype data — not actual college electricity measurements.</b>
        &nbsp;|&nbsp; System demonstrates algorithmic anomaly detection and is not connected to physical meters.
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. SIDEBAR CONTROLS & INTERACTIVE FILTERS
# ==============================================================================
st.sidebar.markdown("### 🎛️ Dashboard Controls")

# Facility Filter
all_bldgs = ["All Buildings"] + sorted(list(df_raw["building"].unique()))
selected_bldg = st.sidebar.selectbox("Campus Facility:", all_bldgs, index=0)

# Severity Filter
severity_options = ["All Severities", "High", "Medium", "Low", "Normal"]
selected_severity = st.sidebar.selectbox("Anomaly Severity:", severity_options, index=0)

# Status Filter
status_options = ["All Records", "Anomalies Only", "Normal Only"]
selected_status = st.sidebar.selectbox("Record Status:", status_options, index=0)

# Date Range Filter
min_dt = df_raw["timestamp"].min().date()
max_dt = df_raw["timestamp"].max().date()

date_filter = st.sidebar.date_input(
    "Date Range:",
    value=[min_dt, max_dt],
    min_value=min_dt,
    max_value=max_dt
)

# Tariff Configuration (₹ / kWh)
st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Tariff Configuration")
tariff_rate = st.sidebar.number_input(
    "Electricity Tariff (₹ / kWh):",
    min_value=1.0,
    max_value=30.0,
    value=8.50,
    step=0.50,
    help="Configurable illustrative utility tariff in ₹/kWh. Not an official college tariff."
)
st.sidebar.caption("ℹ️ Illustrative tariff used for campus cost estimation.")

# ==============================================================================
# 5. DATA FILTERING PIPELINE
# ==============================================================================
df_filtered = df_raw.copy()

if selected_bldg != "All Buildings":
    df_filtered = df_filtered[df_filtered["building"] == selected_bldg]

if selected_severity != "All Severities":
    df_filtered = df_filtered[df_filtered["severity"] == selected_severity]

if selected_status == "Anomalies Only":
    df_filtered = df_filtered[df_filtered["status"] == "Anomaly"]
elif selected_status == "Normal Only":
    df_filtered = df_filtered[df_filtered["status"] == "Normal"]

if isinstance(date_filter, (list, tuple)) and len(date_filter) == 2:
    start_dt = pd.to_datetime(date_filter[0])
    end_dt = pd.to_datetime(date_filter[1]) + pd.Timedelta(days=1)
    df_filtered = df_filtered[(df_filtered["timestamp"] >= start_dt) & (df_filtered["timestamp"] < end_dt)]
    date_str_display = f"{date_filter[0]} to {date_filter[1]}"
else:
    date_str_display = f"{min_dt} to {max_dt}"

# Ensure expected_kwh exists for all calculations
if "expected_kwh" not in df_filtered.columns:
    df_filtered["expected_kwh"] = df_filtered["energy_usage_kwh"]
    df_filtered["kwh_deviation"] = 0.0

# Calculate KPI Aggregates
total_records = len(df_filtered)
total_energy = df_filtered["energy_usage_kwh"].sum() if total_records > 0 else 0.0
avg_hourly = df_filtered["energy_usage_kwh"].mean() if total_records > 0 else 0.0
total_anomalies = (df_filtered["status"] == "Anomaly").sum() if total_records > 0 else 0
high_severity = (df_filtered["severity"] == "High").sum() if total_records > 0 else 0
med_severity = (df_filtered["severity"] == "Medium").sum() if total_records > 0 else 0
low_severity = (df_filtered["severity"] == "Low").sum() if total_records > 0 else 0

total_cost = total_energy * tariff_rate

# Excess Consumption calculation: max(0, actual - expected)
if total_records > 0:
    excess_kwh_series = df_filtered.apply(lambda r: max(0.0, r["energy_usage_kwh"] - r["expected_kwh"]), axis=1)
    total_excess_kwh = excess_kwh_series.sum()
else:
    total_excess_kwh = 0.0
total_excess_cost = total_excess_kwh * tariff_rate

# Filter summary display in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Records Matched:** `{total_records:,}` of `{len(df_raw):,}`")
st.sidebar.markdown(f"**Anomalies:** `{total_anomalies:,}` ({total_anomalies/max(1, total_records)*100:.1f}%)")

# ==============================================================================
# 6. MAIN NAVIGATION TABS (All 8 Academic Sections)
# ==============================================================================
nav_tabs = st.tabs([
    "📊 Overview",
    "📈 Energy Trends",
    "🔍 Anomaly Inspector",
    "🏢 Building Comparison",
    "💰 Cost Analysis",
    "🔮 Forecasting & Baselines",
    "🧠 Model Evaluation",
    "📑 Reports & Export"
])

# ------------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------------------------
with nav_tabs[0]:
    if total_records == 0:
        st.warning("⚠️ No electricity records match the selected filter criteria. Please adjust your filters in the sidebar.")
    else:
        # High Severity Warning Alert Banner
        unack_high = high_severity - len(st.session_state.acknowledged_alerts)
        if high_severity > 0:
            c_alert1, c_alert2 = st.columns([4, 1])
            with c_alert1:
                st.markdown(f"""
                <div class="alert-panel">
                    <h4 style="margin: 0; color: #ef4444;">🚨 High-Severity Alert Center</h4>
                    <p style="margin: 4px 0 0 0; color: #fca5a5; font-size: 0.95rem;">
                        <b>{high_severity:,}</b> critical consumption deviations flagged in the selected scope. 
                        ({max(0, unack_high)} unacknowledged in current browser session).
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with c_alert2:
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                if st.button("✅ Acknowledge Alerts", key="ack_top_alerts", help="Acknowledge current high-severity alerts for this browser session"):
                    st.session_state.acknowledged_alerts.update(
                        df_filtered[df_filtered["severity"] == "High"]["timestamp"].astype(str).tolist()
                    )
                    st.rerun()

        # Top KPI Metric Cards (6 cards in 2 rows of 3, or 6 columns)
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
        
        with kpi_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Energy</div>
                <div class="metric-value">{total_energy:,.0f} <span style="font-size:0.95rem;">kWh</span></div>
                <div class="metric-sub">Across {df_filtered['building'].nunique()} facilities</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Detected Anomalies</div>
                <div class="metric-value" style="color: #ef4444;">{total_anomalies:,}</div>
                <div class="metric-sub">{total_anomalies/max(1, total_records)*100:.1f}% anomaly rate</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">High Severity</div>
                <div class="metric-value" style="color: #dc2626;">{high_severity:,}</div>
                <div class="metric-sub">Require immediate inspection</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Hourly Power</div>
                <div class="metric-value">{avg_hourly:.1f} <span style="font-size:0.95rem;">kWh/h</span></div>
                <div class="metric-sub">Mean facility hourly load</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Est. Cost</div>
                <div class="metric-value" style="color: #38bdf8;">₹{total_cost:,.0f}</div>
                <div class="metric-sub">At ₹{tariff_rate:.2f} / kWh</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Potential Excess Cost</div>
                <div class="metric-value" style="color: #f59e0b;">₹{total_excess_cost:,.0f}</div>
                <div class="metric-sub">{total_excess_kwh:,.0f} kWh above baseline</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Overview Quick Visuals
        ov_col1, ov_col2 = st.columns([3, 2])
        
        with ov_col1:
            # 24-Hour Diurnal Average Curve by Building
            df_diurnal = df_filtered.groupby(["building", "hour"])["energy_usage_kwh"].mean().reset_index()
            fig_diurnal = px.line(
                df_diurnal,
                x="hour",
                y="energy_usage_kwh",
                color="building",
                markers=True,
                title="<b>24-Hour Average Energy Consumption Profile by Facility</b>",
                labels={"hour": "Hour of Day (0–23)", "energy_usage_kwh": "Average Usage (kWh)", "building": "Facility"},
                template="plotly_dark"
            )
            fig_diurnal.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified")
            st.plotly_chart(fig_diurnal, use_container_width=True)
            
        with ov_col2:
            # Facility Share of Total Energy
            df_pie = df_filtered.groupby("building")["energy_usage_kwh"].sum().reset_index()
            fig_pie_bldg = px.pie(
                df_pie,
                names="building",
                values="energy_usage_kwh",
                title="<b>Campus Electricity Consumption Share (%)</b>",
                template="plotly_dark",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_pie_bldg.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_pie_bldg, use_container_width=True)

        st.markdown("""
        <div class="academic-note">
            💡 <b>Academic Note on Campus Energy Profiles:</b> 
            Computer and electronics laboratories exhibit noticeable baselines even during off-peak hours due to networked servers 
            and standby workstations. Classroom blocks, conversely, show near-zero night power consumption but steep spikes 
            during daytime lecture schedules.
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: ENERGY TRENDS
# ------------------------------------------------------------------------------
with nav_tabs[1]:
    st.subheader("📈 Multi-Facility Electricity Consumption & Anomaly Timeline")
    
    if total_records == 0:
        st.warning("⚠️ No data available for the selected filters.")
    else:
        # Time-Series Chart
        # Limit points plotted if dataset is very large to maintain 60fps responsiveness
        max_plot_points = 3500
        if len(df_filtered) > max_plot_points:
            df_time_view = df_filtered.sample(max_plot_points, random_state=42).sort_values("timestamp")
            st.caption(f"ℹ️ Displaying a representative sample of {max_plot_points:,} hourly observations for interactive responsiveness.")
        else:
            df_time_view = df_filtered.sort_values("timestamp")

        fig_ts = px.line(
            df_time_view,
            x="timestamp",
            y="energy_usage_kwh",
            color="building" if selected_bldg == "All Buildings" else None,
            title="Hourly Electricity Consumption Time-Series with Highlighted Anomaly Events",
            labels={"energy_usage_kwh": "Energy Usage (kWh)", "timestamp": "Timestamp", "building": "Facility"},
            template="plotly_dark"
        )
        
        # Overlay Anomaly Markers with Color Coding by Severity
        anoms_in_view = df_time_view[df_time_view["status"] == "Anomaly"]
        
        if len(anoms_in_view) > 0:
            sev_colors = {"High": "#ef4444", "Medium": "#f97316", "Low": "#eab308"}
            for sev, col in sev_colors.items():
                sub_anom = anoms_in_view[anoms_in_view["severity"] == sev]
                if len(sub_anom) > 0:
                    fig_ts.add_trace(go.Scatter(
                        x=sub_anom["timestamp"],
                        y=sub_anom["energy_usage_kwh"],
                        mode="markers",
                        name=f"Anomaly ({sev})",
                        marker=dict(
                            color=col,
                            size=8 if sev == "Low" else (10 if sev == "Medium" else 12),
                            symbol="circle",
                            line=dict(color="#ffffff", width=1)
                        ),
                        hovertext=sub_anom.apply(
                            lambda r: (
                                f"<b>{r['building']}</b><br/>"
                                f"Time: {r['timestamp']}<br/>"
                                f"Usage: {r['energy_usage_kwh']} kWh<br/>"
                                f"Baseline: {r['expected_kwh']} kWh<br/>"
                                f"Severity: <b>{r['severity']}</b> (Score: {r['anomaly_score']:.3f})<br/>"
                                f"Occupancy: {r['occupancy']} | Temp: {r['temperature']}°C"
                            ),
                            axis=1
                        ),
                        hoverinfo="text"
                    ))
                    
        fig_ts.update_layout(height=460, margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Sub-Charts: Hour-of-day boxplot & Weekday vs Weekend Comparison
        c_trend1, c_trend2 = st.columns(2)
        
        with c_trend1:
            st.markdown("#### 🕒 Hour-of-Day Consumption Distribution")
            fig_hour_box = px.box(
                df_filtered,
                x="hour",
                y="energy_usage_kwh",
                color="status",
                title="Hourly Consumption Variance (Normal vs Anomaly)",
                labels={"hour": "Hour of Day", "energy_usage_kwh": "Energy Usage (kWh)", "status": "Classification"},
                color_discrete_map={"Normal": "#38bdf8", "Anomaly": "#ef4444"},
                template="plotly_dark"
            )
            fig_hour_box.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_hour_box, use_container_width=True)
            
        with c_trend2:
            st.markdown("#### 📅 Weekday vs Weekend Load Profile")
            # Map day of week
            day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            df_filtered_copy = df_filtered.copy()
            df_filtered_copy["day_name"] = df_filtered_copy["day_of_week"].map(day_map)
            day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            
            day_agg = df_filtered_copy.groupby(["day_name", "status"])["energy_usage_kwh"].mean().reset_index()
            fig_day = px.bar(
                day_agg,
                x="day_name",
                y="energy_usage_kwh",
                color="status",
                barmode="group",
                category_orders={"day_name": day_order},
                title="Average Consumption by Day of Week",
                labels={"day_name": "Day", "energy_usage_kwh": "Mean kWh", "status": "Status"},
                color_discrete_map={"Normal": "#38bdf8", "Anomaly": "#ef4444"},
                template="plotly_dark",
                text_auto=".1f"
            )
            fig_day.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_day, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: ANOMALY INSPECTOR & FLUCTUATION ANALYSIS
# ------------------------------------------------------------------------------
with nav_tabs[2]:
    st.subheader("🔬 Transparent Anomaly Inspector & Fluctuation Analysis")
    st.markdown("""
    Select any flagged anomaly event from the filter scope to inspect its contextual features,
    transparent rule-based explanation, baseline variance, and recommended maintenance action.
    """)
    
    anoms_df = df_filtered[df_filtered["status"] == "Anomaly"].sort_values("timestamp", ascending=False)
    
    if len(anoms_df) == 0:
        st.info("ℹ️ No anomalies found in the active filter selection. Try widening the building, date range, or severity filters.")
    else:
        # Summary Counters for Current Scope
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Flagged Events", f"{len(anoms_df):,}")
        ac2.metric("High Severity", f"{(anoms_df['severity'] == 'High').sum():,}", delta="Critical", delta_color="inverse")
        ac3.metric("Medium Severity", f"{(anoms_df['severity'] == 'Medium').sum():,}")
        ac4.metric("Low Severity", f"{(anoms_df['severity'] == 'Low').sum():,}")
        
        st.markdown("#### 📋 Detected Anomalies Event Roster")
        
        # Display interactive table of anomalies
        display_cols = [
            "timestamp", "building", "energy_usage_kwh", "expected_kwh",
            "kwh_deviation", "occupancy", "temperature", "anomaly_score", "severity", "explanation"
        ]
        available_cols = [c for c in display_cols if c in anoms_df.columns]
        
        st.dataframe(
            anoms_df[available_cols],
            use_container_width=True,
            hide_index=True,
            height=240
        )
        
        st.markdown("---")
        st.markdown("#### 🎯 Deep-Dive Event Diagnostics")
        
        # Selection Box
        event_options = [
            f"[{row['severity']}] {row['timestamp'].strftime('%Y-%m-%d %H:%M')} | {row['building']} | Usage: {row['energy_usage_kwh']:.1f} kWh (Base: {row.get('expected_kwh', 0):.1f} kWh) | Score: {row['anomaly_score']:.3f}"
            for _, row in anoms_df.head(200).iterrows()
        ]
        
        selected_event_str = st.selectbox(
            "Select an Anomaly Event for Comprehensive Root-Cause Analysis:",
            event_options,
            index=0
        )
        
        if selected_event_str:
            sel_idx = event_options.index(selected_event_str)
            event_row = anoms_df.iloc[sel_idx]
            
            # Diagnostic Card
            sev_badge_color = "#ef4444" if event_row["severity"] == "High" else ("#f97316" if event_row["severity"] == "Medium" else "#eab308")
            
            st.markdown(f"""
            <div class="inspector-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; color:#f8fafc;">🏢 {event_row['building']}</h3>
                    <span style="background:{sev_badge_color}; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; font-size:0.9rem;">
                        {event_row['severity']} Severity Anomaly
                    </span>
                </div>
                <p style="color:#94a3b8; margin:4px 0 16px 0;">Recorded Timestamp: <code>{event_row['timestamp']}</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Actual Energy", f"{event_row['energy_usage_kwh']:.2f} kWh")
            m2.metric("Historical Baseline", f"{event_row.get('expected_kwh', 0):.2f} kWh")
            dev_val = event_row.get('kwh_deviation', 0)
            m3.metric("Deviation (Δ)", f"{dev_val:+.2f} kWh", delta=f"{dev_val:+.1f} kWh", delta_color="inverse")
            m4.metric("Occupancy", f"{event_row['occupancy']} people")
            m5.metric("Anomaly Score", f"{event_row['anomaly_score']:.4f}")

            # Baseline vs Actual Gauge / Bar
            st.markdown("<br>", unsafe_allow_html=True)
            gauge_col, expl_col = st.columns([1, 1])
            
            with gauge_col:
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name="Historical Baseline",
                    x=["Consumption (kWh)"],
                    y=[event_row.get("expected_kwh", 0)],
                    marker_color="#38bdf8",
                    text=[f"{event_row.get('expected_kwh', 0):.1f} kWh"],
                    textposition="auto"
                ))
                fig_comp.add_trace(go.Bar(
                    name="Actual Measured Usage",
                    x=["Consumption (kWh)"],
                    y=[event_row["energy_usage_kwh"]],
                    marker_color=sev_badge_color,
                    text=[f"{event_row['energy_usage_kwh']:.1f} kWh"],
                    textposition="auto"
                ))
                fig_comp.update_layout(
                    barmode="group",
                    title="<b>Actual Usage vs Comparable Historical Baseline</b>",
                    template="plotly_dark",
                    height=280,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
            with expl_col:
                explanation_text = event_row.get("explanation", "Unusual multidimensional consumption pattern detected.")
                action_text = event_row.get("action_recommendation", "Review concurrent building power sub-meters and telemetry logs.")
                
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #334155; padding: 18px; border-radius: 10px; height: 100%;">
                    <h4 style="color: #38bdf8; margin-top: 0;">💡 Rule-Based Contextual Diagnosis:</h4>
                    <p style="font-size: 1.02rem; color: #f1f5f9; font-weight: 600; line-height: 1.4;">
                        "{explanation_text}"
                    </p>
                    <hr style="border-color: #334155; margin: 12px 0;">
                    <h4 style="color: #4ade80; margin-top: 0;">🛠️ Recommended Next Action:</h4>
                    <p style="font-size: 0.95rem; color: #cbd5e1; line-height: 1.4;">
                        <b>{action_text}</b>
                    </p>
                    <p style="font-size: 0.8rem; color: #64748b; margin-top: 10px; margin-bottom: 0;">
                        <i>Note: Isolation Forest provides statistical outlier detection; the explanation is rule-based contextual reasoning to assist facility technicians. Physical inspection is required before concluding hardware faults.</i>
                    </p>
                </div>
                """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 4: BUILDING COMPARISON
# ------------------------------------------------------------------------------
with nav_tabs[3]:
    st.subheader("🏢 Campus Facility Energy Efficiency & Anomaly Comparison")
    
    st.markdown("""
    <div class="academic-note">
        <b>⚖️ Fair Comparison Principle (Academic Governance):</b><br/>
        Campus facilities differ inherently in floor area, specialized lab equipment, server idle power, and student density.
        A higher total consumption in <b>CSE Lab</b> or <b>Seminar Hall</b> does not indicate operational inefficiency.
        Fair auditing requires evaluating occupancy-weighted load, off-hours baseline discipline, and anomaly frequencies.
    </div>
    """, unsafe_allow_html=True)
    
    if total_records == 0:
        st.warning("⚠️ No data available for comparison.")
    else:
        # Grouped summary by building
        bldg_metrics = []
        for b_name, b_df in df_raw.groupby("building"):
            b_total_kwh = b_df["energy_usage_kwh"].sum()
            b_avg_kwh = b_df["energy_usage_kwh"].mean()
            
            # Occupied period average (occupancy > 5)
            occ_df = b_df[b_df["occupancy"] > 5]
            b_occ_avg = occ_df["energy_usage_kwh"].mean() if len(occ_df) > 0 else b_avg_kwh
            
            b_anoms = (b_df["status"] == "Anomaly").sum()
            b_high_anoms = (b_df["severity"] == "High").sum()
            b_cost = b_total_kwh * tariff_rate
            
            if "expected_kwh" in b_df.columns:
                b_excess = b_df.apply(lambda r: max(0.0, r["energy_usage_kwh"] - r["expected_kwh"]), axis=1).sum()
            else:
                b_excess = 0.0
                
            bldg_metrics.append({
                "Facility": b_name,
                "Total Energy (kWh)": round(b_total_kwh, 1),
                "Mean Hourly (kWh/h)": round(b_avg_kwh, 2),
                "Occupied Mean (kWh/h)": round(b_occ_avg, 2),
                "Total Anomalies": int(b_anoms),
                "High Severity": int(b_high_anoms),
                "Estimated Cost (₹)": round(b_cost, 2),
                "Potential Excess (kWh)": round(b_excess, 1)
            })
            
        bldg_summary_df = pd.DataFrame(bldg_metrics)
        
        # Comparative Rankings Table
        st.dataframe(
            bldg_summary_df.sort_values("Total Energy (kWh)", ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("#### 📊 Metric Ranking Visualizer")
        
        rank_metric = st.selectbox(
            "Select Metric to Rank Facilities:",
            [
                "Total Energy (kWh)",
                "Mean Hourly (kWh/h)",
                "Occupied Mean (kWh/h)",
                "Total Anomalies",
                "High Severity",
                "Estimated Cost (₹)",
                "Potential Excess (kWh)"
            ],
            index=0
        )
        
        sorted_bldg_df = bldg_summary_df.sort_values(rank_metric, ascending=True)
        
        fig_rank = px.bar(
            sorted_bldg_df,
            x=rank_metric,
            y="Facility",
            orientation="h",
            title=f"<b>Facility Ranking by {rank_metric}</b>",
            text=rank_metric,
            color=rank_metric,
            color_continuous_scale="Tealgrn" if "Energy" in rank_metric else "OrRd",
            template="plotly_dark"
        )
        fig_rank.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_rank, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: COST ANALYSIS & POTENTIAL WASTAGE ESTIMATOR
# ------------------------------------------------------------------------------
with nav_tabs[4]:
    st.subheader("💰 Campus Electricity Cost & Potential Wastage Estimator")
    
    st.markdown(f"""
    <div class="academic-note">
        <b>📐 Mathematical Calculation Methodology:</b><br/>
        • <b>Total Estimated Cost:</b> Total Energy Consumption (kWh) × Configured Tariff (₹{tariff_rate:.2f}/kWh)<br/>
        • <b>Potential Excess Consumption:</b> $\\sum \\max(0, \\text{{Actual Usage}} - \\text{{Historical Baseline Expected Usage}})$<br/>
        • <b>Potential Excess Cost:</b> Potential Excess Energy (kWh) × Configured Tariff (₹{tariff_rate:.2f}/kWh)<br/>
        <i>Important: An anomaly does not automatically imply energy wastage. Legitimate campus events (such as extended research or weather heatwaves) increase valid demand.</i>
    </div>
    """, unsafe_allow_html=True)
    
    if total_records == 0:
        st.warning("⚠️ No data available for cost analysis.")
    else:
        # Cost Metrics
        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        c_kpi1.metric("Total Energy Billed", f"{total_energy:,.0f} kWh")
        c_kpi2.metric("Total Estimated Cost", f"₹{total_cost:,.2f}")
        c_kpi3.metric("Potential Excess Energy", f"{total_excess_kwh:,.1f} kWh")
        c_kpi4.metric("Potential Excess Cost", f"₹{total_excess_cost:,.2f}", delta=f"{total_excess_cost/max(1.0, total_cost)*100:.1f}% of total bill", delta_color="inverse")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        cost_c1, cost_c2 = st.columns(2)
        
        with cost_c1:
            # Building Cost Breakdown Bar
            cost_by_bldg = df_filtered.groupby("building")["energy_usage_kwh"].sum().reset_index()
            cost_by_bldg["Estimated_Cost_INR"] = cost_by_bldg["energy_usage_kwh"] * tariff_rate
            
            fig_cost_bldg = px.bar(
                cost_by_bldg.sort_values("Estimated_Cost_INR", ascending=False),
                x="building",
                y="Estimated_Cost_INR",
                title=f"<b>Facility-Wise Estimated Electricity Cost (₹)</b> [Tariff: ₹{tariff_rate}/kWh]",
                labels={"building": "Facility", "Estimated_Cost_INR": "Cost (₹)"},
                color="Estimated_Cost_INR",
                color_continuous_scale="Purples",
                template="plotly_dark",
                text_auto=".0f"
            )
            fig_cost_bldg.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_cost_bldg, use_container_width=True)
            
        with cost_c2:
            # Normal vs Excess Cost Split by Building
            excess_by_bldg = []
            for b_name, b_df in df_filtered.groupby("building"):
                b_tot = b_df["energy_usage_kwh"].sum()
                b_exc = b_df.apply(lambda r: max(0.0, r["energy_usage_kwh"] - r["expected_kwh"]), axis=1).sum()
                b_base = max(0.0, b_tot - b_exc)
                excess_by_bldg.append({
                    "Facility": b_name,
                    "Baseline Expected Cost (₹)": b_base * tariff_rate,
                    "Potential Excess Cost (₹)": b_exc * tariff_rate
                })
            df_split = pd.DataFrame(excess_by_bldg)
            
            fig_split = px.bar(
                df_split,
                x="Facility",
                y=["Baseline Expected Cost (₹)", "Potential Excess Cost (₹)"],
                title="<b>Baseline vs Potential Excess Energy Cost Breakdown</b>",
                barmode="stack",
                color_discrete_map={"Baseline Expected Cost (₹)": "#0284c7", "Potential Excess Cost (₹)": "#f59e0b"},
                template="plotly_dark"
            )
            fig_split.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_split, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 6: FORECASTING & BASELINES
# ------------------------------------------------------------------------------
with nav_tabs[5]:
    st.subheader("🔮 Expected Consumption Forecasting & Baseline Analytics")
    
    st.markdown("""
    <div class="academic-note">
        <b>📈 Baseline Estimation Methodology:</b><br/>
        Expected energy consumption represents the empirical median historical demand for the given campus facility,
        hour of day, and weekday/weekend profile. This provides an interpretable benchmark against which anomalies
        and demand spikes are quantified.
    </div>
    """, unsafe_allow_html=True)
    
    if total_records == 0:
        st.warning("⚠️ No data available for baseline analysis.")
    else:
        # Sample for clean timeline view if date range spans months
        forecast_sample = df_filtered.copy()
        if len(forecast_sample) > 500:
            forecast_sample = forecast_sample.head(500).sort_values("timestamp")
            st.caption("ℹ️ Displaying consecutive 500 hourly records to illustrate actual vs expected baseline trajectories.")
        else:
            forecast_sample = forecast_sample.sort_values("timestamp")

        fig_base = go.Figure()
        
        # Expected baseline line
        fig_base.add_trace(go.Scatter(
            x=forecast_sample["timestamp"],
            y=forecast_sample["expected_kwh"],
            mode="lines",
            name="Expected Baseline (kWh)",
            line=dict(color="#38bdf8", width=2, dash="dash")
        ))
        
        # Actual consumption line
        fig_base.add_trace(go.Scatter(
            x=forecast_sample["timestamp"],
            y=forecast_sample["energy_usage_kwh"],
            mode="lines",
            name="Actual Measured Usage (kWh)",
            line=dict(color="#818cf8", width=1.5)
        ))
        
        # Highlight significant deviations
        big_devs = forecast_sample[abs(forecast_sample["kwh_deviation"]) > 15.0]
        if len(big_devs) > 0:
            fig_base.add_trace(go.Scatter(
                x=big_devs["timestamp"],
                y=big_devs["energy_usage_kwh"],
                mode="markers",
                name="Significant Deviation (>15 kWh)",
                marker=dict(color="#f43f5e", size=8, symbol="diamond")
            ))
            
        fig_base.update_layout(
            title="<b>Actual Usage vs Expected Historical Baseline Timeline</b>",
            xaxis_title="Timestamp",
            yaxis_title="Electricity Consumption (kWh)",
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig_base, use_container_width=True)
        
        # Deviation Percentage Distribution
        c_dev1, c_dev2 = st.columns(2)
        with c_dev1:
            fig_dev_hist = px.histogram(
                df_filtered,
                x="kwh_deviation",
                nbins=50,
                color="status",
                title="<b>Distribution of kWh Deviations from Baseline</b>",
                labels={"kwh_deviation": "Deviation from Baseline (kWh)", "status": "Classification"},
                color_discrete_map={"Normal": "#38bdf8", "Anomaly": "#ef4444"},
                template="plotly_dark"
            )
            fig_dev_hist.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_dev_hist, use_container_width=True)
            
        with c_dev2:
            fig_dev_scat = px.scatter(
                df_filtered.sample(min(1200, len(df_filtered)), random_state=42),
                x="expected_kwh",
                y="energy_usage_kwh",
                color="status",
                title="<b>Actual vs Expected Baseline Correlation</b>",
                labels={"expected_kwh": "Expected Baseline (kWh)", "energy_usage_kwh": "Actual Usage (kWh)"},
                color_discrete_map={"Normal": "#38bdf8", "Anomaly": "#ef4444"},
                template="plotly_dark"
            )
            fig_dev_scat.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_dev_scat, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 7: MODEL EVALUATION & TRANSPARENCY
# ------------------------------------------------------------------------------
with nav_tabs[6]:
    st.subheader("🧠 Machine Learning Governance & Model Transparency")
    
    st.markdown("""
    <div class="academic-note">
        <b>🎓 Academic Viva Presentation Summary:</b><br/>
        • <b>Algorithm:</b> Isolation Forest (Unsupervised Ensemble of 150 Isolation Trees)<br/>
        • <b>Key Mechanism:</b> Anomalies are few and quantitatively disparate, requiring fewer random recursive splits to isolate.<br/>
        • <b>Features Used:</b> <code>hour</code>, <code>day_of_week</code>, <code>is_weekend</code>, <code>energy_usage_kwh</code>, <code>temperature</code>, <code>humidity</code>, <code>occupancy</code>.<br/>
        • <b>Data Scaling:</b> <code>StandardScaler</code> applied prior to tree induction.<br/>
        • <b>Contamination Factor:</b> Configured to <code>0.04 (4%)</code> matching estimated physical failure densities.
    </div>
    """, unsafe_allow_html=True)
    
    if metrics_data and "precision" in metrics_data:
        m = metrics_data
        cm = m.get("confusion_matrix", {})
        
        # Metrics KPI Row
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Precision (Synthetic)", f"{m.get('precision', 0):.4f}", help="True Positives / (True Positives + False Positives)")
        e2.metric("Recall (Synthetic)", f"{m.get('recall', 0):.4f}", help="True Positives / (True Positives + False Negatives)")
        e3.metric("F1-Score (Synthetic)", f"{m.get('f1_score', 0):.4f}", help="Harmonic mean of precision and recall")
        e4.metric("Accuracy (Synthetic)", f"{m.get('accuracy', 0):.4f}", help="Overall proportion of correct classifications")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Confusion Matrix Heatmap
        c_cm1, c_cm2 = st.columns([1, 1])
        
        with c_cm1:
            st.markdown("#### 🔢 Confusion Matrix (vs Synthetic Labels)")
            
            cm_z = [
                [cm.get("true_negative", 0), cm.get("false_positive", 0)],
                [cm.get("false_negative", 0), cm.get("true_positive", 0)]
            ]
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_z,
                x=["Predicted Normal", "Predicted Anomaly"],
                y=["Actual Normal", "Actual Anomaly"],
                colorscale="Blues",
                text=[
                    [f"TN: {cm.get('true_negative', 0):,}", f"FP: {cm.get('false_positive', 0):,}"],
                    [f"FN: {cm.get('false_negative', 0):,}", f"TP: {cm.get('true_positive', 0):,}"]
                ],
                texttemplate="%{text}",
                textfont={"size": 14}
            ))
            fig_cm.update_layout(
                title="<b>Confusion Matrix Heatmap</b>",
                template="plotly_dark",
                height=340,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with c_cm2:
            st.markdown("#### ⚖️ Honest Limitations of Synthetic Evaluation")
            st.markdown(f"""
            <div style="background: #1e293b; border: 1px solid #334155; padding: 18px; border-radius: 10px; font-size: 0.9rem; line-height: 1.5;">
                <p><b>1. Unsupervised Operation:</b> Isolation Forest operates without supervision during training. It never receives synthetic labels, ensuring <b>zero data leakage</b>.</p>
                <p><b>2. Evaluation Boundary:</b> Ground-truth labels are synthetic. They benchmark model behavior against specific modeled failure modes (unusual night loads, power surges, phase drops).</p>
                <p><b>3. Real-World Caveat:</b> Unsupervised models will isolate genuine operational variations (such as unscheduled laboratory workshops) as mathematical outliers (False Positives). Physical inspection by facility engineers is essential before diagnosing faults.</p>
                <p style="color: #94a3b8; font-size: 0.82rem; margin-bottom: 0;">
                    <i>{m.get('evaluation_disclaimer', '')}</i>
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Evaluation metrics not loaded. Re-run `python train_model.py` to refresh.")

# ------------------------------------------------------------------------------
# TAB 8: REPORTS & EXPORT
# ------------------------------------------------------------------------------
with nav_tabs[7]:
    st.subheader("📑 Audit Report Generation & Data Export")
    
    st.markdown("""
    Generate and download customized audit reports reflecting the current filter scope,
    selected campus facilities, date ranges, and configured tariff parameters.
    """)
    
    rep_c1, rep_c2, rep_c3 = st.columns(3)
    
    # 1. Filtered Dataset CSV
    with rep_c1:
        st.markdown("#### 📥 Filtered Dataset (CSV)")
        st.write(f"Contains all `{total_records:,}` records matching active filters.")
        csv_filtered_bytes = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv_filtered_bytes,
            file_name="SmartCampus_Filtered_Electricity_Data.csv",
            mime="text/csv",
            key="dl_filtered_csv",
            use_container_width=True
        )
        
    # 2. Anomalies Only CSV
    with rep_c2:
        st.markdown("#### 🚨 Flagged Anomalies (CSV)")
        anom_export = df_filtered[df_filtered["status"] == "Anomaly"]
        st.write(f"Contains `{len(anom_export):,}` flagged anomaly records.")
        csv_anom_bytes = anom_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Anomalies List (CSV)",
            data=csv_anom_bytes,
            file_name="SmartCampus_Detected_Anomalies.csv",
            mime="text/csv",
            key="dl_anom_csv",
            use_container_width=True
        )
        
    # 3. Dynamic PDF Report via ReportLab
    with rep_c3:
        st.markdown("#### 📄 Executive PDF Audit Summary")
        st.write("Publication-grade PDF with KPIs, building breakdown & governance disclaimers.")
        
        try:
            pdf_bytes = report_generator.generate_pdf_report(
                df_filtered=df_filtered,
                tariff=tariff_rate,
                selected_bldg=selected_bldg,
                date_range_str=date_str_display,
                metrics_summary=metrics_data
            )
            
            st.download_button(
                label="📄 Download Executive Audit PDF",
                data=pdf_bytes,
                file_name="SmartCampus_EnergyGuard_Audit_Report.pdf",
                mime="application/pdf",
                key="dl_audit_pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error compiling PDF: {str(e)}")

    st.markdown("---")
    st.markdown("#### 📋 Current Report Parameters Summary")
    st.json({
        "Campus Facility Scope": selected_bldg,
        "Severity Filter": selected_severity,
        "Record Status Filter": selected_status,
        "Date Range": date_str_display,
        "Configured Tariff": f"₹{tariff_rate:.2f} / kWh",
        "Total Filtered Records": total_records,
        "Detected Anomalies": total_anomalies,
        "High Severity Count": high_severity,
        "Total Estimated Cost": f"₹{total_cost:,.2f}"
    })

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding-bottom: 20px;">
    <b>SmartCampus EnergyGuard Academic Prototype</b> • Built with Python, Scikit-learn (Isolation Forest), Plotly & Streamlit<br>
    <i>DISCLAIMER: Simulated prototype data — not actual college electricity measurements. System is not connected to physical meters.</i>
</div>
""", unsafe_allow_html=True)
