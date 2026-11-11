"""
SmartCampus EnergyGuard - Machine Learning Training & Anomaly Detection
========================================================================
DISCLAIMER: Simulated prototype data — not actual college electricity measurements.
Trains an Isolation Forest unsupervised model to isolate multi-dimensional electricity anomalies,
computes historical baseline expectations, assigns transparent severity & rule-based explanations,
and evaluates performance against synthetic ground truth labels without data leakage.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

def compute_historical_baselines(df):
    """
    Computes a robust historical baseline lookup table grouped by
    (building, is_weekend, hour) using median energy consumption.
    This provides an empirical benchmark for expected usage.
    """
    # Use median to resist extreme outlier influence
    baseline_lookup = df.groupby(["building", "is_weekend", "hour"])["energy_usage_kwh"].median().to_dict()
    return baseline_lookup

def assign_explanation_and_action(row):
    """
    Generates transparent, rule-based explanations and actionable recommendations
    by comparing actual measurements against contextual baselines.
    
    Academic Note: Model detection is statistical outlier isolation; explanations
    are rule-assisted hypotheses to guide facility engineers.
    """
    if row["status"] != "Anomaly":
        return (
            "Consumption profile aligns with expected historical operational patterns.",
            "Normal operation. Routine energy monitoring active."
        )
        
    actual = row["energy_usage_kwh"]
    expected = row["expected_kwh"]
    occ = row["occupancy"]
    hour = row["hour"]
    is_weekend = row["is_weekend"]
    
    # 1. Non-working night hours with high load
    if hour in [22, 23, 0, 1, 2, 3, 4, 5] and actual > expected * 1.5:
        explanation = (
            f"Unusually high consumption during non-working hours "
            f"(Actual: {actual:.1f} kWh vs Baseline: {expected:.1f} kWh with 0 occupancy)."
        )
        action = "Inspect facility HVAC timers and zone lighting; verify that centralized equipment was powered down overnight."
        
    # 2. Low occupancy with excessive load
    elif occ <= 5 and actual > expected * 1.6:
        explanation = (
            f"High energy usage despite low recorded occupancy "
            f"(Actual: {actual:.1f} kWh with only {occ} occupants; baseline: {expected:.1f} kWh)."
        )
        action = "Conduct an on-site walkthrough of unoccupied labs/zones to detect idle high-draw appliances or air-conditioning units."
        
    # 3. High occupancy with abnormally low consumption (sensor drop / power phase trip)
    elif occ >= 25 and actual < expected * 0.35:
        explanation = (
            f"Unusually low consumption during an occupied period "
            f"(Actual: {actual:.1f} kWh vs Baseline: {expected:.1f} kWh with {occ} occupants); verify reading or equipment status."
        )
        action = "Inspect sub-meter telemetry and distribution panel for a tripped circuit breaker or phase line failure."
        
    # 4. Unexpected weekend surge
    elif is_weekend == 1 and actual > expected * 1.7:
        explanation = (
            f"Unexpected weekend consumption exceeding normal non-operational baselines "
            f"(Actual: {actual:.1f} kWh vs Weekend Baseline: {expected:.1f} kWh)."
        )
        action = "Audit weekend facility access logs and verify if special academic lab sessions or events were authorized."
        
    # 5. Generic substantial spike
    elif actual > expected * 1.5:
        explanation = (
            f"Consumption is substantially higher than the building's comparable historical usage "
            f"(Actual: {actual:.1f} kWh vs Baseline: {expected:.1f} kWh, deviation: +{(actual - expected):.1f} kWh)."
        )
        action = "Check electrical panel for transient surges, equipment overload, or cooling compressor continuous duty cycles."
        
    # 6. Fallback explanation
    else:
        explanation = (
            f"Unusual multi-dimensional pattern detected (Score: {row['anomaly_score']:.3f}); "
            f"further investigation is recommended."
        )
        action = "Review concurrent environmental factors (ambient temperature, humidity) and sub-meter logs with facility staff."
        
    return explanation, action

def train_and_detect_anomalies(
    input_path="data/energy_data.csv",
    output_csv="data/smartcampus_energy_with_anomalies.csv",
    legacy_csv="detected_anomalies.csv",
    models_dir="models"
):
    """
    Loads simulated dataset, fits StandardScaler and IsolationForest,
    evaluates against synthetic labels, calculates baselines and explanations,
    and serializes models and metrics.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}. Run generate_data.py first.")
        
    print("=" * 75)
    print("SMARTCAMPUS ENERGYGUARD — ML TRAINING & ANOMALY DETECTION PIPELINE")
    print("DISCLAIMER: Simulated prototype data — not actual college electricity measurements.")
    print("=" * 75)
    print(f"Loading energy dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Temporal Feature Engineering
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # 2. Historical Baseline Calculation
    # Compute median usage for each (building, is_weekend, hour)
    baseline_lookup = compute_historical_baselines(df)
    df["expected_kwh"] = df.apply(
        lambda r: round(baseline_lookup.get((r["building"], r["is_weekend"], r["hour"]), r["energy_usage_kwh"]), 2),
        axis=1
    )
    df["kwh_deviation"] = round(df["energy_usage_kwh"] - df["expected_kwh"], 2)
    df["kwh_deviation_pct"] = round(
        ((df["energy_usage_kwh"] - df["expected_kwh"]) / np.maximum(df["expected_kwh"], 0.1)) * 100.0, 1
    )
    
    # 3. Model Training (Strictly WITHOUT Data Leakage)
    # Synthetic anomaly labels are EXCLUDED from training features
    feature_cols = [
        "hour",
        "day_of_week",
        "is_weekend",
        "energy_usage_kwh",
        "temperature",
        "humidity",
        "occupancy"
    ]
    
    X = df[feature_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest Hyperparameters
    # Contamination set to 0.04 (~4%) to align with expected anomaly density
    contamination_rate = 0.04
    n_estimators = 150
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination_rate,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"\nTraining Isolation Forest (n_estimators={n_estimators}, contamination={contamination_rate})...")
    model.fit(X_scaled)
    
    # 4. Predictions & Anomaly Score
    # Scikit-Learn convention: 1 = Normal, -1 = Anomaly
    raw_preds = model.predict(X_scaled)
    
    # decision_function gives negative score for anomalies.
    # Invert so higher positive score = higher abnormality
    scores_raw = -model.decision_function(X_scaled)
    min_s, max_s = scores_raw.min(), scores_raw.max()
    anomaly_scores = (scores_raw - min_s) / (max_s - min_s + 1e-8)
    anomaly_scores = np.round(anomaly_scores, 4)
    
    df["prediction"] = raw_preds
    df["anomaly_score"] = anomaly_scores
    df["status"] = np.where(raw_preds == -1, "Anomaly", "Normal")
    
    # 5. Transparent Severity Classification
    # Score thresholds documented and configurable
    # High: anomaly_score >= 0.72 OR extreme deviation
    # Medium: 0.58 <= anomaly_score < 0.72
    # Low: anomaly_score < 0.58
    def classify_severity(row):
        if row["status"] == "Normal":
            return "Normal"
        score = row["anomaly_score"]
        dev = abs(row["kwh_deviation"])
        if score >= 0.72 or dev >= 25.0:
            return "High"
        elif score >= 0.58 or dev >= 12.0:
            return "Medium"
        else:
            return "Low"
            
    df["severity"] = df.apply(classify_severity, axis=1)
    
    # 6. Contextual Explanations & Action Recommendations
    print("Generating rule-based explanations and actionable investigation steps...")
    explanations_and_actions = df.apply(assign_explanation_and_action, axis=1)
    df["explanation"] = [ea[0] for ea in explanations_and_actions]
    df["action_recommendation"] = [ea[1] for ea in explanations_and_actions]
    
    # Format timestamp for clean CSV persistence
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # 7. Model Evaluation against Synthetic Ground Truth
    has_ground_truth = "is_synthetic_anomaly" in df.columns
    metrics_summary = {}
    
    if has_ground_truth:
        y_true = df["is_synthetic_anomaly"].values
        # Isolation Forest: -1 is anomaly -> 1; 1 is normal -> 0
        y_pred = (df["prediction"] == -1).astype(int)
        
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        
        metrics_summary = {
            "algorithm": "Isolation Forest",
            "features_used": feature_cols,
            "contamination_rate": contamination_rate,
            "n_estimators": n_estimators,
            "total_records": len(df),
            "synthetic_anomalies_injected": int(y_true.sum()),
            "detected_anomalies_by_model": int(y_pred.sum()),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp
            },
            "evaluation_disclaimer": (
                "Metrics reflect detection performance against synthetically injected anomaly patterns only. "
                "Unsupervised Isolation Forest isolates mathematical outliers in feature space and does not "
                "guarantee detection of all unmodeled real-world electrical faults."
            )
        }
        
    # 8. Save Artifacts
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save enriched datasets
    df.to_csv(output_csv, index=False)
    df.to_csv(legacy_csv, index=False)
    print(f"Enriched dataset saved to: {output_csv} (and synced to {legacy_csv})")
    
    # Save model artifact
    model_artifact_path = os.path.join(models_dir, "anomaly_model.pkl")
    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "contamination": contamination_rate,
        "baseline_lookup": baseline_lookup,
        "metrics_summary": metrics_summary,
        "severity_thresholds": {
            "high_score": 0.72,
            "medium_score": 0.58,
            "high_deviation_kwh": 25.0,
            "medium_deviation_kwh": 12.0
        }
    }
    joblib.dump(model_bundle, model_artifact_path)
    print(f"Model bundle serialized to: {model_artifact_path}")
    
    # Save evaluation metrics JSON
    metrics_path = os.path.join(models_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Evaluation metrics report saved to: {metrics_path}")
    
    # 9. Print Viva-Ready Summary
    total_records = len(df)
    total_anomalies = (df["status"] == "Anomaly").sum()
    high_sev = (df["severity"] == "High").sum()
    med_sev = (df["severity"] == "Medium").sum()
    low_sev = (df["severity"] == "Low").sum()
    
    print("\n" + "=" * 75)
    print("DETECTION & EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Total Hourly Records Processed : {total_records:,}")
    print(f"Total Anomalies Flagged        : {total_anomalies:,} ({total_anomalies / total_records * 100:.2f}%)")
    print(f" - High Severity (Critical)    : {high_sev:,}")
    print(f" - Medium Severity (Moderate)  : {med_sev:,}")
    print(f" - Low Severity (Minor Drift)  : {low_sev:,}")
    
    if has_ground_truth:
        print("\nSynthetic Ground Truth Validation:")
        print(f" - Precision                   : {metrics_summary['precision']:.4f}")
        print(f" - Recall                      : {metrics_summary['recall']:.4f}")
        print(f" - F1-Score                    : {metrics_summary['f1_score']:.4f}")
        print(f" - Confusion Matrix            : TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"\nAcademic Transparency Note:\n{metrics_summary['evaluation_disclaimer']}")
    print("=" * 75)
    
    return df, metrics_summary

if __name__ == "__main__":
    train_and_detect_anomalies()
