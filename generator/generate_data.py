"""
CareSignals Healthcare Outreach Intelligence System
Synthetic Patient Data Generator
Generates realistic de-identified patient data for POC development
"""

import sqlite3
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

random.seed(42)
np.random.seed(42)

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare.db')

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CONDITIONS = [
    'Type 2 Diabetes', 'Hypertension', 'Heart Disease', 'COPD',
    'Asthma', 'Obesity', 'Depression', 'Hyperlipidemia',
    'Chronic Kidney Disease', 'Osteoarthritis'
]

MEDICATIONS = {
    'Type 2 Diabetes':        ['Metformin', 'Insulin', 'Glipizide', 'Jardiance'],
    'Hypertension':           ['Lisinopril', 'Amlodipine', 'Losartan', 'Metoprolol'],
    'Heart Disease':          ['Aspirin', 'Atorvastatin', 'Carvedilol', 'Warfarin'],
    'COPD':                   ['Albuterol', 'Tiotropium', 'Fluticasone', 'Salmeterol'],
    'Asthma':                 ['Albuterol', 'Budesonide', 'Montelukast', 'Prednisone'],
    'Obesity':                ['Phentermine', 'Orlistat', 'Wegovy', 'Contrave'],
    'Depression':             ['Sertraline', 'Fluoxetine', 'Escitalopram', 'Bupropion'],
    'Hyperlipidemia':         ['Atorvastatin', 'Rosuvastatin', 'Simvastatin', 'Ezetimibe'],
    'Chronic Kidney Disease': ['Lisinopril', 'Erythropoietin', 'Calcitriol', 'Sevelamer'],
    'Osteoarthritis':         ['Ibuprofen', 'Naproxen', 'Celecoxib', 'Acetaminophen'],
}

PREVENTIVE_CARE = [
    'Annual Wellness Visit', 'Flu Shot', 'Mammogram', 'Colonoscopy',
    'Blood Pressure Check', 'HbA1c Test', 'Cholesterol Panel',
    'Eye Exam', 'Dental Cleaning', 'Bone Density Scan'
]

CITIES = [
    ('Omaha', 'NE'), ('Lincoln', 'NE'), ('Bellevue', 'NE'),
    ('Grand Island', 'NE'), ('Kearney', 'NE'), ('Fremont', 'NE'),
    ('Hastings', 'NE'), ('Norfolk', 'NE'), ('Columbus', 'NE'), ('Papillion', 'NE')
]

ENGAGEMENT_CHANNELS = ['SMS', 'Email', 'Push Notification', 'Phone Call']

MESSAGE_TYPES = [
    'Medication Reminder', 'Appointment Reminder', 'Preventive Care Nudge',
    'Flu Shot Campaign', 'Annual Wellness Reminder', 'Lab Result Follow-up',
    'Chronic Disease Management', 'Post-Surgical Recovery', 'Mental Health Check-in'
]


def create_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.executescript("""
    DROP TABLE IF EXISTS patients;
    DROP TABLE IF EXISTS conditions;
    DROP TABLE IF EXISTS medications;
    DROP TABLE IF EXISTS appointments;
    DROP TABLE IF EXISTS preventive_care;
    DROP TABLE IF EXISTS outreach_messages;
    DROP TABLE IF EXISTS engagement_events;
    DROP TABLE IF EXISTS risk_scores;
    DROP TABLE IF EXISTS broadcast_campaigns;

    CREATE TABLE patients (
        patient_id      TEXT PRIMARY KEY,
        age             INTEGER,
        gender          TEXT,
        city            TEXT,
        state           TEXT,
        insurance_plan  TEXT,
        member_since    TEXT,
        last_visit_date TEXT,
        days_since_visit INTEGER,
        num_conditions  INTEGER,
        num_medications INTEGER,
        medication_adherence_pct REAL,
        missed_appointments INTEGER,
        er_visits_last_year INTEGER,
        hospitalizations_last_year INTEGER,
        engagement_score REAL,
        preferred_channel TEXT,
        risk_level      TEXT,
        risk_score      REAL,
        spoilage_flag   INTEGER
    );

    CREATE TABLE conditions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id  TEXT,
        condition   TEXT,
        diagnosed_date TEXT,
        severity    TEXT,
        controlled  INTEGER
    );

    CREATE TABLE medications (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id      TEXT,
        medication_name TEXT,
        condition       TEXT,
        prescribed_date TEXT,
        adherence_pct   REAL,
        refill_due_date TEXT,
        days_until_refill INTEGER
    );

    CREATE TABLE appointments (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id      TEXT,
        appointment_type TEXT,
        scheduled_date  TEXT,
        status          TEXT,
        days_until_appt INTEGER
    );

    CREATE TABLE preventive_care (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id      TEXT,
        care_type       TEXT,
        last_completed  TEXT,
        days_overdue    INTEGER,
        recommended     INTEGER
    );

    CREATE TABLE outreach_messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id      TEXT,
        message_type    TEXT,
        channel         TEXT,
        message_text    TEXT,
        sent_date       TEXT,
        opened          INTEGER,
        clicked         INTEGER,
        responded       INTEGER,
        campaign_id     TEXT
    );

    CREATE TABLE engagement_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id      TEXT,
        event_type      TEXT,
        event_date      TEXT,
        channel         TEXT,
        outcome         TEXT
    );

    CREATE TABLE risk_scores (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id      TEXT,
        score_date      TEXT,
        risk_score      REAL,
        risk_level      TEXT,
        readmission_prob REAL,
        missed_care_prob REAL,
        nonadherence_prob REAL,
        model_version   TEXT
    );

    CREATE TABLE broadcast_campaigns (
        campaign_id     TEXT PRIMARY KEY,
        campaign_name   TEXT,
        target_condition TEXT,
        target_age_min  INTEGER,
        target_age_max  INTEGER,
        message_template TEXT,
        sent_date       TEXT,
        total_sent      INTEGER,
        open_rate       REAL,
        response_rate   REAL
    );
    """)
    conn.commit()
    return conn


def generate_patients(n=3000):
    patients = []
    for i in range(n):
        pid = f"PAT{str(i+1).zfill(5)}"
        age = int(np.random.normal(55, 18))
        age = max(18, min(90, age))
        gender = random.choice(['Male', 'Female', 'Female', 'Male', 'Female'])  # slight female skew
        city, state = random.choice(CITIES)
        plan = random.choice(['PPO Gold', 'PPO Silver', 'HMO Basic', 'PPO Platinum', 'Medicare Advantage'])
        member_since = (datetime.now() - timedelta(days=random.randint(180, 3650))).strftime('%Y-%m-%d')

        # Health complexity
        num_conditions = np.random.choice([1, 2, 3, 4, 5], p=[0.30, 0.30, 0.20, 0.12, 0.08])
        num_medications = num_conditions + random.randint(0, 2)

        # Adherence — correlated with age and num conditions
        base_adherence = 85 - (num_conditions * 3) + (age > 65) * 5
        medication_adherence = max(20, min(100, base_adherence + np.random.normal(0, 10)))

        # Visit history
        days_since_visit = random.randint(1, 730)
        missed_appointments = random.randint(0, 5)
        er_visits = np.random.choice([0, 1, 2, 3], p=[0.55, 0.25, 0.13, 0.07])
        hospitalizations = np.random.choice([0, 1, 2], p=[0.70, 0.22, 0.08])

        # Engagement
        engagement_score = max(0, min(100,
            100 - (days_since_visit / 7) - (missed_appointments * 5)
            + (medication_adherence * 0.2) - (er_visits * 8)
        ))

        preferred_channel = random.choice(ENGAGEMENT_CHANNELS)

        # Risk scoring
        risk_score = (
            (num_conditions * 8) +
            (max(0, 100 - medication_adherence) * 0.3) +
            (min(days_since_visit, 365) / 365 * 20) +
            (missed_appointments * 5) +
            (er_visits * 12) +
            (hospitalizations * 15) +
            (age > 65) * 10 +
            np.random.normal(0, 5)
        )
        risk_score = max(0, min(100, risk_score))

        if risk_score >= 60:
            risk_level = 'High'
        elif risk_score >= 35:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        # Missed care flag (target variable) — strongly correlated with features
        missed_care_prob = (
            0.02 +
            (num_conditions * 0.07) +
            (max(0, 100 - medication_adherence) / 100 * 0.40) +
            (min(days_since_visit, 365) / 365 * 0.30) +
            (missed_appointments / 5 * 0.25) +
            (er_visits / 3 * 0.20) +
            (hospitalizations * 0.15) +
            (age > 65) * 0.08
        )
        missed_care_flag = int(random.random() < min(0.97, missed_care_prob))

        last_visit = (datetime.now() - timedelta(days=days_since_visit)).strftime('%Y-%m-%d')

        patients.append({
            'patient_id': pid,
            'age': age,
            'gender': gender,
            'city': city,
            'state': state,
            'insurance_plan': plan,
            'member_since': member_since,
            'last_visit_date': last_visit,
            'days_since_visit': days_since_visit,
            'num_conditions': num_conditions,
            'num_medications': num_medications,
            'medication_adherence_pct': round(medication_adherence, 1),
            'missed_appointments': missed_appointments,
            'er_visits_last_year': er_visits,
            'hospitalizations_last_year': hospitalizations,
            'engagement_score': round(engagement_score, 1),
            'preferred_channel': preferred_channel,
            'risk_level': risk_level,
            'risk_score': round(risk_score, 1),
            'spoilage_flag': missed_care_flag
        })
    return patients


def generate_conditions(patients):
    rows = []
    for p in patients:
        pid = p['patient_id']
        n = p['num_conditions']
        chosen = random.sample(CONDITIONS, min(n, len(CONDITIONS)))
        for cond in chosen:
            diagnosed = (datetime.now() - timedelta(days=random.randint(180, 3650))).strftime('%Y-%m-%d')
            severity = random.choice(['Mild', 'Moderate', 'Moderate', 'Severe'])
            controlled = random.choice([1, 1, 0])
            rows.append((pid, cond, diagnosed, severity, controlled))
    return rows


def generate_medications(patients):
    rows = []
    for p in patients:
        pid = p['patient_id']
        n = p['num_medications']
        conds = random.sample(CONDITIONS, min(p['num_conditions'], len(CONDITIONS)))
        count = 0
        for cond in conds:
            if count >= n:
                break
            meds = MEDICATIONS.get(cond, [])
            if meds:
                med = random.choice(meds)
                prescribed = (datetime.now() - timedelta(days=random.randint(30, 1000))).strftime('%Y-%m-%d')
                adherence = max(20, min(100, p['medication_adherence_pct'] + np.random.normal(0, 8)))
                refill_days = random.randint(-30, 90)  # negative = overdue
                refill_date = (datetime.now() + timedelta(days=refill_days)).strftime('%Y-%m-%d')
                rows.append((pid, med, cond, prescribed, round(adherence, 1), refill_date, refill_days))
                count += 1
    return rows


def generate_appointments(patients):
    rows = []
    appt_types = ['Primary Care', 'Specialist', 'Lab Work', 'Follow-up', 'Telehealth', 'Cardiology', 'Endocrinology']
    statuses = ['Scheduled', 'Completed', 'No-Show', 'Cancelled', 'Rescheduled']
    status_weights = [0.35, 0.40, 0.10, 0.10, 0.05]
    for p in patients:
        pid = p['patient_id']
        n_appts = random.randint(1, 4)
        for _ in range(n_appts):
            days_offset = random.randint(-180, 90)
            appt_date = (datetime.now() + timedelta(days=days_offset)).strftime('%Y-%m-%d')
            status = np.random.choice(statuses, p=status_weights)
            if days_offset > 0:
                status = 'Scheduled'
            rows.append((pid, random.choice(appt_types), appt_date, status, days_offset))
    return rows


def generate_preventive_care(patients):
    rows = []
    for p in patients:
        pid = p['patient_id']
        n_care = random.randint(2, 5)
        chosen = random.sample(PREVENTIVE_CARE, n_care)
        for care in chosen:
            days_ago = random.randint(0, 730)
            last_done = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            # Annual items overdue if > 365 days
            overdue = max(0, days_ago - 365)
            recommended = 1 if overdue > 0 or random.random() < 0.3 else 0
            rows.append((pid, care, last_done, overdue, recommended))
    return rows


def generate_outreach_messages(patients):
    rows = []
    campaigns = ['CAMP001', 'CAMP002', 'CAMP003', 'CAMP004', None]
    for p in patients:
        pid = p['patient_id']
        n_msgs = random.randint(1, 6)
        for _ in range(n_msgs):
            msg_type = random.choice(MESSAGE_TYPES)
            channel = p['preferred_channel'] if random.random() < 0.7 else random.choice(ENGAGEMENT_CHANNELS)
            days_ago = random.randint(0, 180)
            sent_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            opened = int(random.random() < 0.45)
            clicked = int(opened and random.random() < 0.35)
            responded = int(clicked and random.random() < 0.40)
            campaign_id = random.choice(campaigns)
            msg_text = f"[{msg_type}] Personalized health communication for patient {pid}"
            rows.append((pid, msg_type, channel, msg_text, sent_date, opened, clicked, responded, campaign_id))
    return rows


def generate_risk_scores(patients):
    rows = []
    for p in patients:
        pid = p['patient_id']
        score_date = datetime.now().strftime('%Y-%m-%d')
        rs = p['risk_score']
        readmission_prob = min(0.95, rs / 100 * 0.8 + np.random.normal(0, 0.05))
        missed_care_prob = min(0.95, rs / 100 * 0.7 + np.random.normal(0, 0.05))
        nonadherence_prob = min(0.95, (100 - p['medication_adherence_pct']) / 100 * 0.9)
        rows.append((pid, score_date, rs, p['risk_level'],
                     round(max(0, readmission_prob), 3),
                     round(max(0, missed_care_prob), 3),
                     round(max(0, nonadherence_prob), 3),
                     'v1.0'))
    return rows


def generate_broadcast_campaigns():
    campaigns = [
        ('CAMP001', 'Fall Flu Shot Campaign', 'All Members', 18, 90,
         'Time to get your flu shot! Schedule your free vaccination at any BCBSNE network pharmacy.',
         '2025-09-15', 2800, 0.42, 0.18),
        ('CAMP002', 'Diabetes Management Program', 'Type 2 Diabetes', 30, 80,
         'Managing diabetes is easier with support. Join our free Diabetes Care Program today.',
         '2025-10-01', 650, 0.51, 0.24),
        ('CAMP003', 'Annual Wellness Visit Reminder', 'All Members', 40, 90,
         'Your annual wellness visit is covered 100%. Schedule yours today — it only takes 30 minutes.',
         '2025-10-15', 1900, 0.38, 0.21),
        ('CAMP004', 'Heart Health Awareness', 'Heart Disease', 45, 85,
         'February is Heart Month. Your heart health matters — schedule your cardiac screening today.',
         '2025-11-01', 420, 0.47, 0.19),
    ]
    return campaigns


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    print("Creating database...")
    conn = create_database()
    c = conn.cursor()

    print("Generating 3,000 patients...")
    patients = generate_patients(3000)
    c.executemany("""
        INSERT INTO patients VALUES (
            :patient_id, :age, :gender, :city, :state, :insurance_plan,
            :member_since, :last_visit_date, :days_since_visit,
            :num_conditions, :num_medications, :medication_adherence_pct,
            :missed_appointments, :er_visits_last_year, :hospitalizations_last_year,
            :engagement_score, :preferred_channel, :risk_level, :risk_score, :spoilage_flag
        )
    """, patients)

    print("Generating conditions...")
    cond_rows = generate_conditions(patients)
    c.executemany("INSERT INTO conditions (patient_id, condition, diagnosed_date, severity, controlled) VALUES (?,?,?,?,?)", cond_rows)

    print("Generating medications...")
    med_rows = generate_medications(patients)
    c.executemany("INSERT INTO medications (patient_id, medication_name, condition, prescribed_date, adherence_pct, refill_due_date, days_until_refill) VALUES (?,?,?,?,?,?,?)", med_rows)

    print("Generating appointments...")
    appt_rows = generate_appointments(patients)
    c.executemany("INSERT INTO appointments (patient_id, appointment_type, scheduled_date, status, days_until_appt) VALUES (?,?,?,?,?)", appt_rows)

    print("Generating preventive care records...")
    prev_rows = generate_preventive_care(patients)
    c.executemany("INSERT INTO preventive_care (patient_id, care_type, last_completed, days_overdue, recommended) VALUES (?,?,?,?,?)", prev_rows)

    print("Generating outreach messages...")
    msg_rows = generate_outreach_messages(patients)
    c.executemany("INSERT INTO outreach_messages (patient_id, message_type, channel, message_text, sent_date, opened, clicked, responded, campaign_id) VALUES (?,?,?,?,?,?,?,?,?)", msg_rows)

    print("Generating risk scores...")
    risk_rows = generate_risk_scores(patients)
    c.executemany("INSERT INTO risk_scores (patient_id, score_date, risk_score, risk_level, readmission_prob, missed_care_prob, nonadherence_prob, model_version) VALUES (?,?,?,?,?,?,?,?)", risk_rows)

    print("Generating broadcast campaigns...")
    camp_rows = generate_broadcast_campaigns()
    c.executemany("INSERT INTO broadcast_campaigns VALUES (?,?,?,?,?,?,?,?,?,?)", camp_rows)

    conn.commit()
    conn.close()

    # Summary
    conn2 = sqlite3.connect(DB_PATH)
    for table in ['patients', 'conditions', 'medications', 'appointments', 'preventive_care', 'outreach_messages', 'risk_scores', 'broadcast_campaigns']:
        count = pd.read_sql_query(f"SELECT COUNT(*) as n FROM {table}", conn2)['n'][0]
        print(f"  {table}: {count:,} records")
    conn2.close()
    print(f"\nDatabase saved to: {DB_PATH}")


if __name__ == '__main__':
    main()
