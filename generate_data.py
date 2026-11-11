"""
SmartCampus EnergyGuard - Dataset Generator
============================================
DISCLAIMER: Simulated prototype data — not actual college electricity measurements.
This script generates a realistic simulated college electricity dataset for ML anomaly detection.
Never claim that the system is connected to actual college electricity meters.
"""

import os
import numpy as np
import pandas as pd

# Set fixed random seed for reproducibility
np.random.seed(42)

def generate_college_electricity_data(output_path="data/energy_data.csv", total_days=180):
    """
    Generates realistic hourly electricity consumption data across seven campus facilities
    over approximately 6 months (~30,240 records total, > 3,000 records).
    
    Injects ~3.5% controlled synthetic anomalies with ground truth labels saved
    strictly for model evaluation (not used as training features).
    """
    
    # 7 Campus Facilities
    buildings = [
        "CSE Lab 1",
        "CSE Lab 2",
        "ECE Lab",
        "Library",
        "Admin Block",
        "Classroom Block",
        "Seminar Hall"
    ]
    
    # Generate date range: 180 days (~6 months) of hourly timestamps
    start_date = pd.Timestamp("2026-01-01 00:00:00")
    date_range = pd.date_range(start=start_date, periods=total_days * 24, freq="h")
    
    records = []
    
    for dt in date_range:
        hour = dt.hour
        day_of_week = dt.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Diurnal temperature cycle (18°C to 40°C depending on hour and seasonal drift)
        # Day progression drift over the 6 months (Jan to June warms up)
        day_index = (dt - start_date).days
        seasonal_offset = (day_index / total_days) * 6.0  # +0°C in Jan up to +6°C in June
        
        base_temp = 23.0 + seasonal_offset + 7.0 * np.sin(np.pi * (hour - 8) / 12) + np.random.normal(0, 1.2)
        temperature = round(float(np.clip(base_temp, 16.0, 44.0)), 1)
        
        # Diurnal humidity cycle (30% to 85%)
        base_hum = 65.0 - (seasonal_offset * 1.5) - 18.0 * np.sin(np.pi * (hour - 8) / 12) + np.random.normal(0, 2.5)
        humidity = round(float(np.clip(base_hum, 25.0, 92.0)), 1)
        
        for bldg in buildings:
            # Determine realistic base occupancy by facility type & time
            if is_weekend:
                if bldg in ["Library", "Seminar Hall"]:
                    # Occasional student study sessions or weekend workshops
                    occ = np.random.randint(0, 30) if 10 <= hour <= 16 else 0
                elif bldg in ["CSE Lab 1", "CSE Lab 2", "ECE Lab"]:
                    # Research scholars / project work
                    occ = np.random.randint(0, 10) if 10 <= hour <= 15 else 0
                else:
                    occ = np.random.randint(0, 4) if 9 <= hour <= 13 else 0
            else:  # Weekday
                if 8 <= hour <= 17:
                    if bldg in ["CSE Lab 1", "CSE Lab 2"]:
                        occ = np.random.randint(40, 80)
                    elif bldg == "ECE Lab":
                        occ = np.random.randint(35, 75)
                    elif bldg == "Classroom Block":
                        occ = np.random.randint(90, 240)
                    elif bldg == "Library":
                        occ = np.random.randint(45, 130)
                    elif bldg == "Admin Block":
                        occ = np.random.randint(25, 55)
                    elif bldg == "Seminar Hall":
                        # Scheduled guest lectures or department events
                        occ = np.random.randint(110, 260) if hour in [10, 11, 14, 15] else np.random.randint(0, 20)
                elif 18 <= hour <= 21:
                    if bldg == "Library":
                        occ = np.random.randint(25, 65)
                    elif bldg in ["CSE Lab 1", "CSE Lab 2"]:
                        occ = np.random.randint(10, 35)
                    else:
                        occ = np.random.randint(0, 12)
                else:  # Night time (22 PM - 7 AM)
                    occ = 0
            
            # Facility-specific load characteristics (Base load in kWh + per occupant load in kWh)
            if bldg in ["CSE Lab 1", "CSE Lab 2"]:
                base_load = 5.5  # Lab servers, network switches, idle desktop standby
                per_person_load = 0.28  # Desktops + monitors + local AC
            elif bldg == "ECE Lab":
                base_load = 6.0  # Bench power supplies, oscilloscopes, fume hoods, soldering stations
                per_person_load = 0.32
            elif bldg == "Library":
                base_load = 4.2  # Automated catalog servers, continuous lighting, central ventilation
                per_person_load = 0.12  # Reading lamps, laptops
            elif bldg == "Admin Block":
                base_load = 3.5  # Office servers, biometric systems, hallway lights
                per_person_load = 0.16  # PCs, printers, cabin AC
            elif bldg == "Classroom Block":
                base_load = 2.0  # Hallway lighting, water coolers
                per_person_load = 0.08  # Ceiling fans, LED tube lights, projectors
            else:  # Seminar Hall
                base_load = 2.8  # PA standby, staging controls
                per_person_load = 0.22  # High-tonnage central chiller, AV system, stage spotlights
            
            # Cooling load multiplier driven by ambient temperature
            if 8 <= hour <= 18:
                temp_load_factor = max(0.0, (temperature - 26.0) * 0.45)
            else:
                temp_load_factor = max(0.0, (temperature - 28.0) * 0.12)
            
            # Compute normal electricity consumption (kWh)
            normal_usage = base_load + (occ * per_person_load) + temp_load_factor + np.random.normal(0, 0.75)
            normal_usage = round(max(1.5, float(normal_usage)), 2)
            
            records.append({
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "building": bldg,
                "energy_usage_kwh": normal_usage,
                "temperature": temperature,
                "humidity": humidity,
                "occupancy": int(occ),
                "is_synthetic_anomaly": 0,
                "synthetic_anomaly_type": "None"
            })
            
    df = pd.DataFrame(records)
    
    # Inject ~3.5% realistic synthetic anomalies
    # Types:
    # 1. Night Usage Spike (HVAC/lights forgotten on with 0 occupancy)
    # 2. Equipment Power Surge (Sudden massive spike during daytime)
    # 3. Weekend Ghost Load (Substantial consumption during non-operational weekend)
    # 4. Occupied Phase Drop (Abnormally low consumption despite high occupancy; phase trip or meter drop)
    
    num_rows = len(df)
    target_anomalies = int(num_rows * 0.035)
    
    # Select candidate indices based on contextual criteria for realistic anomalies
    anomaly_indices = np.random.choice(df.index, size=target_anomalies, replace=False)
    
    for idx in anomaly_indices:
        row = df.loc[idx]
        bldg = row["building"]
        curr_kwh = row["energy_usage_kwh"]
        dt = pd.to_datetime(row["timestamp"])
        hour = dt.hour
        is_weekend = dt.dayofweek >= 5
        
        # Select contextual anomaly type
        if hour in [22, 23, 0, 1, 2, 3, 4, 5]:
            anomaly_type = "Night Usage Spike"
            # High consumption overnight while occupancy is zero
            new_kwh = round(float(np.random.uniform(38.0, 72.0)), 2)
            df.at[idx, "energy_usage_kwh"] = new_kwh
            df.at[idx, "occupancy"] = 0
            df.at[idx, "is_synthetic_anomaly"] = 1
            df.at[idx, "synthetic_anomaly_type"] = anomaly_type
            
        elif is_weekend:
            anomaly_type = "Weekend Ghost Load"
            # High energy usage on weekend with near-zero occupancy
            new_kwh = round(float(np.random.uniform(32.0, 58.0)), 2)
            df.at[idx, "energy_usage_kwh"] = new_kwh
            df.at[idx, "occupancy"] = int(np.random.randint(0, 3))
            df.at[idx, "is_synthetic_anomaly"] = 1
            df.at[idx, "synthetic_anomaly_type"] = anomaly_type
            
        else:
            # 50% chance of surge, 50% chance of phase drop during day
            if np.random.rand() > 0.45 and row["occupancy"] >= 25:
                anomaly_type = "Occupied Phase Drop"
                # Heavy occupancy but anomalous tiny energy reading (meter fault / phase line trip)
                new_kwh = round(float(np.random.uniform(0.2, 1.2)), 2)
                df.at[idx, "energy_usage_kwh"] = new_kwh
                df.at[idx, "is_synthetic_anomaly"] = 1
                df.at[idx, "synthetic_anomaly_type"] = anomaly_type
            else:
                anomaly_type = "Equipment Power Surge"
                # Massive daytime spike (e.g. short circuit test, equipment runaway)
                new_kwh = round(float(curr_kwh * np.random.uniform(2.8, 4.6)), 2)
                df.at[idx, "energy_usage_kwh"] = new_kwh
                df.at[idx, "is_synthetic_anomaly"] = 1
                df.at[idx, "synthetic_anomaly_type"] = anomaly_type
                
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    injected_count = df["is_synthetic_anomaly"].sum()
    
    print("=" * 75)
    print("SMARTCAMPUS ENERGYGUARD — DATASET GENERATION COMPLETE")
    print("DISCLAIMER: Simulated prototype data — not actual college electricity measurements.")
    print("=" * 75)
    print(f"Total Hourly Records Generated : {len(df):,}")
    print(f"Campus Facilities Represented  : {df['building'].nunique()} buildings ({', '.join(df['building'].unique())})")
    print(f"Date Range                     : {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Injected Synthetic Anomalies   : {injected_count:,} ({injected_count / len(df) * 100:.2f}%)")
    print("\nAnomaly Breakdown by Type:")
    for atype, cnt in df["synthetic_anomaly_type"].value_counts().items():
        print(f" - {atype:24s}: {cnt:,}")
    print(f"\nDataset successfully saved to : {output_path}")
    print("=" * 75)
    
    return df

if __name__ == "__main__":
    generate_college_electricity_data()
