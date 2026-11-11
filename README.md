# SmartCampus EnergyGuard
## ML-Based Electricity Usage Anomaly Detection System

> ⚠️ **IMPORTANT DATASET DISCLAIMER**:  
> **Simulated prototype data — not actual college electricity measurements.**

---

## 📌 Problem Statement
Educational institutions operate numerous facilities (computer labs, electronics labs, classrooms, libraries, administrative offices, and seminar halls) with varying daily energy footprints. Unexpected electricity spikes, equipment left running overnight, phase power trips, or faulty HVAC units often go unnoticed until high utility bills arrive or equipment breaks down. Manual tracking of power consumption across multiple facilities is inefficient and non-scalable.

## 🎯 Project Objective
**SmartCampus EnergyGuard** is a prototype Machine Learning application designed to continuously monitor hourly electricity consumption across various college facilities and automatically detect unusual/anomalous electricity usage patterns using unsupervised anomaly detection techniques.

## 💡 Proposed Solution
The system uses the **Isolation Forest** unsupervised machine learning algorithm to isolate abnormal consumption events based on multi-dimensional context (hour of day, day of week, occupancy, ambient temperature, humidity, and energy usage in kWh). Anomalies are flagged, categorized by severity (Low, Medium, High), and presented via an interactive, real-time Streamlit dashboard.

---

## 📊 Dataset Description
The dataset contains **30,240 hourly records** (180 days across 7 facilities) simulated locally using Python with realistic campus patterns:
- **Higher daytime usage** (8:00 AM - 6:00 PM) matching college operational hours.
- **Reduced nighttime baseline** (6:00 PM - 8:00 AM).
- **Occupancy-dependent consumption** in computer & electronics labs (high PC and equipment load).
- **Diurnal temperature & humidity variance** influencing cooling loads.
- **Lower weekend baseline consumption**.
- **Deliberate synthetic anomaly events (~3.5%)** such as overnight HVAC usage with 0 occupancy, extreme equipment overload, weekend ghost loads, and daytime power sensor drops.

### Dataset Columns:
1. `timestamp`: Hourly timestamp (`YYYY-MM-DD HH:MM:SS`)
2. `building`: Campus facility name
3. `energy_usage_kwh`: Electricity consumption in Kilowatt-hours
4. `temperature`: Ambient temperature in °C
5. `humidity`: Relative humidity percentage (%)
6. `occupancy`: Number of people present in the facility

---

## 🛠️ Technologies Used
- **Language**: Python 3.9+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (`IsolationForest`, `StandardScaler`)
- **Interactive Data Visualization**: Plotly Express & Plotly Graph Objects
- **Dashboard UI**: Streamlit
- **Model Serialization**: Joblib

---

## 🤖 ML Algorithm: Isolation Forest
### How Isolation Forest Works (For Viva Explanation):
1. **Unsupervised Learning**: Isolation Forest does not rely on pre-labeled anomaly data during training.
2. **Isolation Principle**: Anomalies are few and quantitatively different from normal data points. Therefore, in random feature space partitioning (decision trees), anomalous points require **fewer splits** to isolate than normal cluster points.
3. **Anomaly Score Calculation**:
   - The tree depth required to isolate a sample measures its normality.
   - Short path lengths indicate high probability of being an anomaly.
   - The raw decision output is converted into an normalized **Anomaly Score** $[0.0, 1.0]$.
4. **Severity Classification**:
   - `Anomaly Score >= 0.70` $\rightarrow$ **High Severity**
   - `0.55 <= Anomaly Score < 0.70` $\rightarrow$ **Medium Severity**
   - `Anomaly Score < 0.55` $\rightarrow$ **Low Severity**

> **System Recommendation Output**:  
> *"Unusual electricity consumption detected. Further investigation is recommended."*

---

## 🔄 Project Workflow
```text
Simulated Energy Dataset (generate_data.py)
              ↓
      Data Preprocessing
              ↓
     Feature Engineering (hour, day, weekend, kWh, temp, hum, occupancy)
              ↓
    Isolation Forest Training (train_model.py)
              ↓
    Anomaly & Severity Classification
              ↓
 Streamlit Dashboard (app.py)
```

---

## 🚀 How to Install & Run

### 1. Clone / Open Project Directory
```bash
cd smartpower
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Generate Simulated Electricity Dataset
```bash
python generate_data.py
```
*Output*: Generates `data/energy_data.csv`.

### 4. Train ML Model & Detect Anomalies
```bash
python train_model.py
```
*Output*: Generates `detected_anomalies.csv` and `models/anomaly_model.pkl`.

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📈 Expected Dashboard Features
- **Key Metrics Overview**: Total Energy (kWh), Total Anomalies Detected, High Severity Anomalies, Average Usage.
- **Interactive Filters**: Facility selector, severity filter, custom date range picker.
- **Visualizations**:
  - Time-series line chart with red anomaly markers.
  - Facility-wise consumption comparison bar chart.
  - Anomaly severity distribution pie chart & occupancy scatter plot.
- **Detailed Anomaly Inspector**: Interactive dropdown to select individual anomaly instances and inspect building, timestamp, energy usage, climate parameters, anomaly score, and system alert recommendations.

---

## 🔮 Future Scope
- Integration with real-time **IoT Smart Energy Meters** (ESP32 / Modbus / MQTT).
- Automated **Email & SMS Alert Notifications** via AWS SNS / Twilio.
- **Electricity Cost Estimation & Utility Bill Forecasting**.
- Automated **Building Energy-Efficiency Audits & Carbon Footprint Reports**.
- Deploying on cloud platforms (Streamlit Community Cloud / AWS / Azure).
