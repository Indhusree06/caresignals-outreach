"""
CareSignals Healthcare Outreach Intelligence System
ML Risk Model Training — 5 classifiers for patient risk prediction
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

DB_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare.db')
SAVE_DIR = os.path.dirname(__file__)

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            p.patient_id,
            p.age,
            p.gender,
            p.insurance_plan,
            p.days_since_visit,
            p.num_conditions,
            p.num_medications,
            p.medication_adherence_pct,
            p.missed_appointments,
            p.er_visits_last_year,
            p.hospitalizations_last_year,
            p.engagement_score,
            p.preferred_channel,
            p.spoilage_flag as missed_care_flag,
            rs.readmission_prob,
            rs.missed_care_prob,
            rs.nonadherence_prob
        FROM patients p
        LEFT JOIN risk_scores rs ON p.patient_id = rs.patient_id
    """, conn)
    conn.close()
    return df

def engineer_features(df):
    # Cast numeric columns (SQLite may return bytes)
    num_cols = ['age', 'days_since_visit', 'num_conditions', 'num_medications',
                'medication_adherence_pct', 'missed_appointments', 'er_visits_last_year',
                'hospitalizations_last_year', 'engagement_score',
                'readmission_prob', 'missed_care_prob', 'nonadherence_prob']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Encode categoricals
    le_gender   = LabelEncoder()
    le_plan     = LabelEncoder()
    le_channel  = LabelEncoder()

    df['gender_enc']  = le_gender.fit_transform(df['gender'])
    df['plan_enc']    = le_plan.fit_transform(df['insurance_plan'])
    df['channel_enc'] = le_channel.fit_transform(df['preferred_channel'])

    # Derived features
    df['age_group']           = pd.cut(df['age'], bins=[0,30,45,60,75,100], labels=[0,1,2,3,4]).astype(int)
    df['high_complexity']     = (df['num_conditions'] >= 3).astype(int)
    df['low_adherence']       = (df['medication_adherence_pct'] < 70).astype(int)
    df['long_gap']            = (df['days_since_visit'] > 180).astype(int)
    df['frequent_er']         = (df['er_visits_last_year'] >= 2).astype(int)
    df['adherence_x_conditions'] = df['medication_adherence_pct'] * df['num_conditions']
    df['visit_gap_x_missed']  = df['days_since_visit'] * df['missed_appointments']
    # Add probability features from risk scoring
    df['readmission_prob']    = df['readmission_prob']
    df['missed_care_prob']    = df['missed_care_prob']
    df['nonadherence_prob']   = df['nonadherence_prob']

    features = [
        'age', 'gender_enc', 'plan_enc', 'channel_enc',
        'days_since_visit', 'num_conditions', 'num_medications',
        'medication_adherence_pct', 'missed_appointments',
        'er_visits_last_year', 'hospitalizations_last_year',
        'engagement_score', 'age_group', 'high_complexity',
        'low_adherence', 'long_gap', 'frequent_er',
        'adherence_x_conditions', 'visit_gap_x_missed',
        'readmission_prob', 'missed_care_prob', 'nonadherence_prob'
    ]

    encoders = {
        'gender': le_gender,
        'insurance_plan': le_plan,
        'preferred_channel': le_channel
    }

    return df, features, encoders

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost':             XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                                             random_state=42, eval_metric='logloss', verbosity=0),
        'LightGBM':            LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                                              random_state=42, verbose=-1),
    }

    results = []
    trained = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_proba)

        results.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1, 'AUC-ROC': auc})
        trained[name] = model
        print(f"    Accuracy={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    return pd.DataFrame(results).sort_values('AUC-ROC', ascending=False), trained

def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} patients loaded")

    print("Engineering features...")
    df, features, encoders = engineer_features(df)

    X = df[features]
    y = df['missed_care_flag']

    print(f"  Class distribution: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training 5 models...")
    results_df, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\nModel Results:")
    print(results_df.to_string(index=False))

    # Best model = LightGBM or XGBoost (highest AUC)
    best_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}")

    # Feature importance from best tree model
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.DataFrame({'feature': features, 'importance': best_model.feature_importances_})
        fi = fi.sort_values('importance', ascending=False)
        joblib.dump(fi, os.path.join(SAVE_DIR, 'feature_importance.pkl'))
        print("Feature importance saved")

    # Save artifacts
    joblib.dump(best_model,  os.path.join(SAVE_DIR, 'risk_model.pkl'))
    joblib.dump(encoders,    os.path.join(SAVE_DIR, 'encoders.pkl'))
    joblib.dump(features,    os.path.join(SAVE_DIR, 'feature_list.pkl'))
    joblib.dump(results_df,  os.path.join(SAVE_DIR, 'model_results.pkl'))
    joblib.dump(trained_models, os.path.join(SAVE_DIR, 'all_models.pkl'))

    print("\nAll artifacts saved to models/")
    print("Done!")

if __name__ == '__main__':
    main()
