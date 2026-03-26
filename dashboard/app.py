"""
CareSignals Healthcare Patient Outreach Intelligence System
Streamlit Dashboard — 5 Pages
Sponsor: Blue Cross Blue Shield of Nebraska (POC)
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime, timedelta
from openai import OpenAI

# ── PATH SETUP ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH    = os.path.join(BASE_DIR, 'data', 'healthcare.db')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareSignals — Healthcare Outreach Intelligence",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f1117;
        color: #e8eaf0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
    }
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 16px;
    }
    [data-testid="metric-container"] label {
        color: #8b92a5 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #2563eb;
    }
    /* Risk badges */
    .badge-high   { background:#7f1d1d; color:#fca5a5; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-medium { background:#78350f; color:#fcd34d; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-low    { background:#14532d; color:#86efac; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    /* Message card */
    .msg-card {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
        line-height: 1.6;
        color: #e8eaf0;
    }
    /* Buttons */
    .stButton > button {
        background: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
    }
    .stButton > button:hover { background: #1d4ed8; }
    /* Suggested question buttons */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: #1a1f2e;
        color: #93c5fd;
        border: 1px solid #2a2f3e;
        font-size: 0.82rem;
        padding: 0.4rem 0.8rem;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: #2563eb;
        color: #ffffff;
    }
    /* Sidebar nav */
    .stRadio > label { color: #8b92a5 !important; font-size: 0.8rem; }
    .stRadio [data-testid="stMarkdownContainer"] p { color: #e8eaf0 !important; }
    /* Tables */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    /* Divider */
    hr { border-color: #2a2f3e; }
    /* Expander */
    .streamlit-expanderHeader { color: #8b92a5 !important; }
</style>
""", unsafe_allow_html=True)

# ── OPENAI CLIENT ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if api_key:
        return OpenAI(api_key=api_key)
    return None

# ── DATABASE ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def query(sql, params=()):
    conn = get_db()
    return pd.read_sql_query(sql, conn, params=params)

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        risk_model   = joblib.load(os.path.join(MODEL_DIR, 'risk_model.pkl'))
        encoders     = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
        feature_list = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))
        model_results= joblib.load(os.path.join(MODEL_DIR, 'model_results.pkl'))
        fi           = joblib.load(os.path.join(MODEL_DIR, 'feature_importance.pkl'))
        return risk_model, encoders, feature_list, model_results, fi
    except Exception as e:
        return None, None, None, None, None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## CareSignals")
    st.markdown("*Healthcare Outreach Intelligence*")
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("", [
        "Patient Overview",
        "Patient Detail",
        "Analytics",
        "Outreach Generator",
        "Model Performance"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Filters**")
    risk_filter = st.multiselect("Risk Level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    st.markdown("---")
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("<small style='color:#4b5563'>Sponsored by BCBSNE<br>UNO AI-CCORE POC</small>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PATIENT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Patient Overview":
    st.markdown('<div class="section-header">Patient Population Overview</div>', unsafe_allow_html=True)
    st.markdown("Real-time view of member health risk across the BCBSNE network.")

    # KPI row
    df_kpi = query("SELECT risk_level, COUNT(*) as n FROM patients GROUP BY risk_level")
    total  = query("SELECT COUNT(*) as n FROM patients")['n'][0]
    high   = df_kpi[df_kpi['risk_level']=='High']['n'].sum() if 'High' in df_kpi['risk_level'].values else 0
    med    = df_kpi[df_kpi['risk_level']=='Medium']['n'].sum() if 'Medium' in df_kpi['risk_level'].values else 0
    low    = df_kpi[df_kpi['risk_level']=='Low']['n'].sum() if 'Low' in df_kpi['risk_level'].values else 0
    avg_adh= query("SELECT AVG(medication_adherence_pct) as a FROM patients")['a'][0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Members",       f"{total:,}")
    c2.metric("High Risk",           f"{high:,}",  delta=f"{high/total*100:.1f}%", delta_color="inverse")
    c3.metric("Medium Risk",         f"{med:,}",   delta=f"{med/total*100:.1f}%",  delta_color="off")
    c4.metric("Low Risk",            f"{low:,}",   delta=f"{low/total*100:.1f}%",  delta_color="normal")
    c5.metric("Avg Med Adherence",   f"{avg_adh:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    # Risk distribution donut
    with col1:
        st.markdown('<div class="section-header">Risk Level Distribution</div>', unsafe_allow_html=True)
        fig_donut = px.pie(
            df_kpi, names='risk_level', values='n',
            color='risk_level',
            color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'},
            hole=0.55
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8eaf0', legend=dict(font=dict(color='#e8eaf0')),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Risk by age group
    with col2:
        st.markdown('<div class="section-header">Risk by Age Group</div>', unsafe_allow_html=True)
        df_age = query("""
            SELECT
                CASE
                    WHEN age < 30 THEN '18-29'
                    WHEN age < 45 THEN '30-44'
                    WHEN age < 60 THEN '45-59'
                    WHEN age < 75 THEN '60-74'
                    ELSE '75+'
                END as age_group,
                risk_level,
                COUNT(*) as n
            FROM patients
            GROUP BY age_group, risk_level
        """)
        fig_age = px.bar(
            df_age, x='age_group', y='n', color='risk_level',
            color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'},
            barmode='stack'
        )
        fig_age.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8eaf0', xaxis=dict(gridcolor='#2a2f3e'),
            yaxis=dict(gridcolor='#2a2f3e'), legend=dict(font=dict(color='#e8eaf0')),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    # Top conditions
    with col3:
        st.markdown('<div class="section-header">Top Chronic Conditions</div>', unsafe_allow_html=True)
        df_cond = query("""
            SELECT condition, COUNT(*) as patients
            FROM conditions
            GROUP BY condition
            ORDER BY patients DESC
            LIMIT 10
        """)
        fig_cond = px.bar(
            df_cond.sort_values('patients'), x='patients', y='condition',
            orientation='h', color='patients',
            color_continuous_scale='Blues'
        )
        fig_cond.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8eaf0', xaxis=dict(gridcolor='#2a2f3e'),
            yaxis=dict(gridcolor='#2a2f3e'), showlegend=False,
            coloraxis_showscale=False, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_cond, use_container_width=True)

    # Engagement by channel
    with col4:
        st.markdown('<div class="section-header">Outreach Channel Performance</div>', unsafe_allow_html=True)
        df_chan = query("""
            SELECT channel,
                   COUNT(*) as sent,
                   ROUND(AVG(opened)*100,1) as open_rate,
                   ROUND(AVG(responded)*100,1) as response_rate
            FROM outreach_messages
            GROUP BY channel
        """)
        fig_chan = go.Figure()
        fig_chan.add_trace(go.Bar(name='Open Rate %',     x=df_chan['channel'], y=df_chan['open_rate'],     marker_color='#3b82f6'))
        fig_chan.add_trace(go.Bar(name='Response Rate %', x=df_chan['channel'], y=df_chan['response_rate'], marker_color='#22c55e'))
        fig_chan.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8eaf0', xaxis=dict(gridcolor='#2a2f3e'),
            yaxis=dict(gridcolor='#2a2f3e'), legend=dict(font=dict(color='#e8eaf0')),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_chan, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">High-Risk Members Requiring Outreach</div>', unsafe_allow_html=True)
    df_high = query("""
        SELECT patient_id, age, gender, city,
               num_conditions, medication_adherence_pct,
               days_since_visit, missed_appointments,
               er_visits_last_year, risk_score, risk_level
        FROM patients
        WHERE risk_level = 'High'
        ORDER BY risk_score DESC
        LIMIT 50
    """)
    st.dataframe(df_high, use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PATIENT DETAIL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Patient Detail":
    st.markdown('<div class="section-header">Patient Detail View</div>', unsafe_allow_html=True)

    col_search, col_btn = st.columns([3, 1])
    with col_search:
        pid_input = st.text_input("Enter Patient ID (e.g. PAT00001)", value="PAT00001")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("Load Patient")

    if search_btn or pid_input:
        pid = pid_input.strip().upper()
        df_p = query("SELECT * FROM patients WHERE patient_id = ?", (pid,))

        if df_p.empty:
            st.error(f"Patient {pid} not found.")
        else:
            p = df_p.iloc[0]
            rl = p['risk_level']
            badge_class = f"badge-{rl.lower()}"

            # Header
            st.markdown(f"""
            <div style='background:#1a1f2e; border:1px solid #2a2f3e; border-radius:10px; padding:20px; margin:10px 0'>
                <h2 style='color:#fff; margin:0'>{p['patient_id']} &nbsp;
                <span class='{badge_class}'>{rl} Risk</span></h2>
                <p style='color:#8b92a5; margin:6px 0 0 0'>
                    {p['gender']}, Age {p['age']} &nbsp;|&nbsp; {p['city']}, {p['state']}
                    &nbsp;|&nbsp; Plan: {p['insurance_plan']}
                    &nbsp;|&nbsp; Member since {p['member_since']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # KPIs
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Risk Score",         f"{p['risk_score']:.1f}/100")
            c2.metric("Med Adherence",      f"{p['medication_adherence_pct']:.0f}%")
            c3.metric("Days Since Visit",   f"{p['days_since_visit']}")
            c4.metric("ER Visits (1yr)",    f"{p['er_visits_last_year']}")
            c5.metric("Missed Appts",       f"{p['missed_appointments']}")

            st.markdown("---")
            col1, col2 = st.columns(2)

            # Conditions
            with col1:
                st.markdown('<div class="section-header">Active Conditions</div>', unsafe_allow_html=True)
                df_cond = query("SELECT condition, severity, controlled, diagnosed_date FROM conditions WHERE patient_id=?", (pid,))
                if not df_cond.empty:
                    df_cond['controlled'] = df_cond['controlled'].apply(lambda x: "Yes" if x else "No")
                    st.dataframe(df_cond, use_container_width=True, hide_index=True)

            # Medications
            with col2:
                st.markdown('<div class="section-header">Current Medications</div>', unsafe_allow_html=True)
                df_med = query("""
                    SELECT medication_name, condition, adherence_pct,
                           refill_due_date, days_until_refill
                    FROM medications WHERE patient_id=?
                """, (pid,))
                if not df_med.empty:
                    st.dataframe(df_med, use_container_width=True, hide_index=True)

            st.markdown("---")
            col3, col4 = st.columns(2)

            # Upcoming appointments
            with col3:
                st.markdown('<div class="section-header">Appointments</div>', unsafe_allow_html=True)
                df_appt = query("""
                    SELECT appointment_type, scheduled_date, status, days_until_appt
                    FROM appointments WHERE patient_id=? ORDER BY scheduled_date
                """, (pid,))
                if not df_appt.empty:
                    st.dataframe(df_appt, use_container_width=True, hide_index=True)

            # Preventive care gaps
            with col4:
                st.markdown('<div class="section-header">Preventive Care Gaps</div>', unsafe_allow_html=True)
                df_prev = query("""
                    SELECT care_type, last_completed, days_overdue, recommended
                    FROM preventive_care WHERE patient_id=? AND recommended=1
                    ORDER BY days_overdue DESC
                """, (pid,))
                if not df_prev.empty:
                    df_prev['recommended'] = df_prev['recommended'].apply(lambda x: "Yes" if x else "No")
                    st.dataframe(df_prev, use_container_width=True, hide_index=True)

            # Risk gauge
            st.markdown("---")
            st.markdown('<div class="section-header">Risk Score Gauge</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(p['risk_score']),
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#8b92a5'},
                    'bar': {'color': '#ef4444' if rl=='High' else '#f59e0b' if rl=='Medium' else '#22c55e'},
                    'steps': [
                        {'range': [0, 35],  'color': '#14532d'},
                        {'range': [35, 60], 'color': '#78350f'},
                        {'range': [60, 100],'color': '#7f1d1d'},
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 3}, 'value': float(p['risk_score'])}
                },
                title={'text': "Patient Risk Score", 'font': {'color': '#e8eaf0'}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font_color='#e8eaf0',
                height=280, margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Outreach history
            st.markdown("---")
            st.markdown('<div class="section-header">Outreach History</div>', unsafe_allow_html=True)
            df_msg = query("""
                SELECT message_type, channel, sent_date, opened, clicked, responded
                FROM outreach_messages WHERE patient_id=? ORDER BY sent_date DESC LIMIT 10
            """, (pid,))
            if not df_msg.empty:
                df_msg['opened']    = df_msg['opened'].apply(lambda x: "Yes" if x else "No")
                df_msg['clicked']   = df_msg['clicked'].apply(lambda x: "Yes" if x else "No")
                df_msg['responded'] = df_msg['responded'].apply(lambda x: "Yes" if x else "No")
                st.dataframe(df_msg, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.markdown('<div class="section-header">Population Analytics</div>', unsafe_allow_html=True)
    st.markdown("Insights on patient engagement, outreach effectiveness, and health trends.")

    col1, col2 = st.columns(2)

    # Medication adherence distribution
    with col1:
        st.markdown('<div class="section-header">Medication Adherence Distribution</div>', unsafe_allow_html=True)
        df_adh = query("SELECT medication_adherence_pct, risk_level FROM patients")
        fig_adh = px.histogram(
            df_adh, x='medication_adherence_pct', color='risk_level',
            color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'},
            nbins=30, barmode='overlay', opacity=0.7
        )
        fig_adh.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8eaf0', xaxis=dict(gridcolor='#2a2f3e'),
            yaxis=dict(gridcolor='#2a2f3e'), legend=dict(font=dict(color='#e8eaf0')),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_adh, use_container_width=True)

    # Days since last visit
    with col2:
        st.markdown('<div class="section-header">Days Since Last Visit by Risk Level</div>', unsafe_allow_html=True)
        df_visit = query("SELECT days_since_visit, risk_level FROM patients")
        fig_box = px.box(
            df_visit, x='risk_level', y='days_since_visit', color='risk_level',
            color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'}
        )
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8eaf0', xaxis=dict(gridcolor='#2a2f3e'),
            yaxis=dict(gridcolor='#2a2f3e'), showlegend=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    # Outreach message type performance
    with col3:
        st.markdown('<div class="section-header">Message Type Effectiveness</div>', unsafe_allow_html=True)
        df_mtype = query("""
            SELECT message_type,
                   COUNT(*) as sent,
                   ROUND(AVG(opened)*100,1) as open_rate,
                   ROUND(AVG(responded)*100,1) as response_rate
            FROM outreach_messages
            GROUP BY message_type
            ORDER BY response_rate DESC
        """)
        fig_mtype = px.scatter(
            df_mtype, x='open_rate', y='response_rate',
            size='sent', text='message_type',
            color='response_rate', color_continuous_scale='Blues',
            labels={'open_rate':'Open Rate (%)', 'response_rate':'Response Rate (%)'}
        )
        fig_mtype.update_traces(textposition='top center', textfont_color='#e8eaf0')
        fig_mtype.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8eaf0', xaxis=dict(gridcolor='#2a2f3e'),
            yaxis=dict(gridcolor='#2a2f3e'), coloraxis_showscale=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_mtype, use_container_width=True)

    # Preventive care gap analysis
    with col4:
        st.markdown('<div class="section-header">Preventive Care Gaps by Type</div>', unsafe_allow_html=True)
        df_gaps = query("""
            SELECT care_type,
                   COUNT(*) as patients_overdue,
                   ROUND(AVG(days_overdue),0) as avg_days_overdue
            FROM preventive_care
            WHERE days_overdue > 0
            GROUP BY care_type
            ORDER BY patients_overdue DESC
        """)
        fig_gaps = px.bar(
            df_gaps.sort_values('patients_overdue'),
            x='patients_overdue', y='care_type',
            orientation='h', color='avg_days_overdue',
            color_continuous_scale='Reds',
            labels={'patients_overdue':'Patients Overdue','avg_days_overdue':'Avg Days Overdue'}
        )
        fig_gaps.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8eaf0', xaxis=dict(gridcolor='#2a2f3e'),
            yaxis=dict(gridcolor='#2a2f3e'), coloraxis_showscale=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_gaps, use_container_width=True)

    st.markdown("---")
    col5, col6 = st.columns(2)

    # ER visits vs risk score scatter
    with col5:
        st.markdown('<div class="section-header">ER Visits vs Risk Score</div>', unsafe_allow_html=True)
        df_er = query("SELECT er_visits_last_year, risk_score, risk_level, num_conditions FROM patients")
        fig_er = px.scatter(
            df_er, x='er_visits_last_year', y='risk_score',
            color='risk_level', size='num_conditions',
            color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'},
            opacity=0.6
        )
        fig_er.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8eaf0', xaxis=dict(gridcolor='#2a2f3e'),
            yaxis=dict(gridcolor='#2a2f3e'), legend=dict(font=dict(color='#e8eaf0')),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_er, use_container_width=True)

    # Insurance plan distribution
    with col6:
        st.markdown('<div class="section-header">Risk Distribution by Insurance Plan</div>', unsafe_allow_html=True)
        df_plan = query("""
            SELECT insurance_plan, risk_level, COUNT(*) as n
            FROM patients GROUP BY insurance_plan, risk_level
        """)
        fig_plan = px.bar(
            df_plan, x='insurance_plan', y='n', color='risk_level',
            color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'},
            barmode='stack'
        )
        fig_plan.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8eaf0', xaxis=dict(gridcolor='#2a2f3e', tickangle=-20),
            yaxis=dict(gridcolor='#2a2f3e'), legend=dict(font=dict(color='#e8eaf0')),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_plan, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Broadcast Campaign Performance</div>', unsafe_allow_html=True)
    df_camp = query("SELECT campaign_name, target_condition, total_sent, open_rate, response_rate FROM broadcast_campaigns")
    df_camp['open_rate']     = (df_camp['open_rate'] * 100).round(1).astype(str) + '%'
    df_camp['response_rate'] = (df_camp['response_rate'] * 100).round(1).astype(str) + '%'
    st.dataframe(df_camp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — OUTREACH GENERATOR (RAG + PERSONALIZATION AGENT)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Outreach Generator":
    st.markdown('<div class="section-header">Personalized Outreach Generator</div>', unsafe_allow_html=True)
    st.markdown("Select a patient to generate a personalized, context-aware outreach message using the Personalization Agent.")

    # DB schema for AI
    DB_SCHEMA = """
    Tables:
    - patients(patient_id, age, gender, city, state, insurance_plan, days_since_visit,
               num_conditions, num_medications, medication_adherence_pct, missed_appointments,
               er_visits_last_year, hospitalizations_last_year, engagement_score,
               preferred_channel, risk_level, risk_score)
    - conditions(patient_id, condition, severity, controlled)
    - medications(patient_id, medication_name, condition, adherence_pct, days_until_refill)
    - appointments(patient_id, appointment_type, scheduled_date, status, days_until_appt)
    - preventive_care(patient_id, care_type, last_completed, days_overdue, recommended)
    - outreach_messages(patient_id, message_type, channel, sent_date, opened, clicked, responded)
    - broadcast_campaigns(campaign_id, campaign_name, target_condition, total_sent, open_rate, response_rate)
    """

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("**Select Patient**")
        df_patients = query("""
            SELECT patient_id, age, gender, risk_level, risk_score,
                   num_conditions, medication_adherence_pct, preferred_channel
            FROM patients ORDER BY risk_score DESC LIMIT 100
        """)
        pid_select = st.selectbox(
            "Patient ID (sorted by risk score)",
            df_patients['patient_id'].tolist(),
            format_func=lambda x: f"{x} — {df_patients[df_patients['patient_id']==x]['risk_level'].values[0]} Risk (Score: {df_patients[df_patients['patient_id']==x]['risk_score'].values[0]:.0f})"
        )

        msg_type = st.selectbox("Message Type", [
            "Personalized Health Journey",
            "Medication Adherence Reminder",
            "Appointment Reminder",
            "Preventive Care Nudge",
            "Chronic Disease Management",
            "Post-Surgical Recovery",
            "Mental Health Check-in",
            "Annual Wellness Reminder"
        ])

        channel = st.selectbox("Delivery Channel", ["SMS", "Email", "Push Notification", "Phone Call"])
        tone    = st.selectbox("Tone", ["Warm and Supportive", "Professional and Direct", "Motivational", "Gentle Reminder"])

    with col_right:
        st.markdown("**Suggested Questions**")
        suggested = [
            "Which patients are overdue for flu shots?",
            "Who has the lowest medication adherence?",
            "How many high-risk patients haven't visited in 6 months?",
            "What is the most effective outreach channel?",
            "Which conditions have the highest missed care rate?"
        ]
        if 'analyst_q' not in st.session_state:
            st.session_state['analyst_q'] = ""
        for q in suggested:
            if st.button(q, key=f"sq_{q}"):
                st.session_state['analyst_q'] = q

        st.markdown("**Ask the Population Analyst**")
        analyst_q = st.text_input("Ask a question about the patient population", value=st.session_state.get('analyst_q',''))

        if st.button("Run Analysis") and analyst_q:
            client = get_openai_client()
            if not client:
                st.warning("OpenAI API key not configured. Add OPENAI_API_KEY to Streamlit Secrets.")
            else:
                with st.spinner("Analyzing population data..."):
                    try:
                        sql_resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": f"""You are a healthcare data analyst. Generate a single valid SQLite SQL query to answer the user's question.
Schema: {DB_SCHEMA}
Rules:
- Return ONLY the SQL query, no explanation, no markdown
- Use exact column names from the schema
- For joins between patients and conditions/medications, use patient_id
- LIMIT results to 20 rows maximum"""},
                                {"role": "user", "content": analyst_q}
                            ]
                        )
                        sql = sql_resp.choices[0].message.content.strip().strip('```sql').strip('```').strip()
                        conn = get_db()
                        df_result = pd.read_sql_query(sql, conn)

                        answer_resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a healthcare analytics assistant. Answer the question based on the data provided. Be concise and specific. Use numbers from the data."},
                                {"role": "user", "content": f"Question: {analyst_q}\n\nData:\n{df_result.to_string(index=False)}"}
                            ]
                        )
                        answer = answer_resp.choices[0].message.content

                        st.markdown(f"**Question:** {analyst_q}")
                        st.markdown(f'<div class="msg-card">{answer}</div>', unsafe_allow_html=True)
                        with st.expander("View Data & SQL"):
                            st.code(sql, language='sql')
                            st.dataframe(df_result, use_container_width=True)
                    except Exception as e:
                        st.error(f"Analysis error: {e}")

    st.markdown("---")
    st.markdown('<div class="section-header">Generate Personalized Message</div>', unsafe_allow_html=True)

    if st.button("Generate Outreach Message", type="primary"):
        # Gather patient context
        p = query("SELECT * FROM patients WHERE patient_id=?", (pid_select,)).iloc[0]
        conds = query("SELECT condition, severity FROM conditions WHERE patient_id=?", (pid_select,))
        meds  = query("SELECT medication_name, adherence_pct, days_until_refill FROM medications WHERE patient_id=?", (pid_select,))
        prev  = query("SELECT care_type, days_overdue FROM preventive_care WHERE patient_id=? AND recommended=1", (pid_select,))
        appts = query("SELECT appointment_type, scheduled_date, status FROM appointments WHERE patient_id=? AND days_until_appt > 0", (pid_select,))

        context = f"""
Patient Profile:
- ID: {p['patient_id']}, Age: {p['age']}, Gender: {p['gender']}, City: {p['city']}, {p['state']}
- Insurance Plan: {p['insurance_plan']}
- Risk Level: {p['risk_level']} (Score: {p['risk_score']:.1f}/100)
- Days since last visit: {p['days_since_visit']}
- Medication adherence: {p['medication_adherence_pct']:.0f}%
- Missed appointments: {p['missed_appointments']}
- ER visits last year: {p['er_visits_last_year']}
- Preferred channel: {p['preferred_channel']}

Active Conditions: {', '.join(conds['condition'].tolist()) if not conds.empty else 'None recorded'}

Medications with low adherence: {', '.join(meds[meds['adherence_pct']<70]['medication_name'].tolist()) if not meds.empty else 'None'}

Preventive care overdue: {', '.join(prev['care_type'].tolist()) if not prev.empty else 'None'}

Upcoming appointments: {', '.join(appts['appointment_type'].tolist()) if not appts.empty else 'None scheduled'}
"""

        client = get_openai_client()
        if not client:
            st.warning("OpenAI API key not configured. Add OPENAI_API_KEY to Streamlit Secrets.")
            # Show a sample message without API
            sample = f"""Dear Member,

We noticed it has been {p['days_since_visit']} days since your last visit. As a valued BCBSNE member managing {p['num_conditions']} health condition(s), staying connected with your care team is important.

Your current medication adherence is {p['medication_adherence_pct']:.0f}%. {"We encourage you to refill your prescriptions soon." if p['medication_adherence_pct'] < 80 else "Keep up the great work with your medications!"}

{"We also see you have some preventive care items due. " + ", ".join(prev['care_type'].tolist()[:2]) + " are recommended for you." if not prev.empty else ""}

Please call your care coordinator or schedule a visit at your convenience.

— BCBSNE Care Team"""
            st.markdown(f'<div class="msg-card">{sample}</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Personalization Agent generating message..."):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": f"""You are a healthcare outreach specialist for Blue Cross Blue Shield of Nebraska.
Generate a personalized {msg_type} message for delivery via {channel}.
Tone: {tone}
Rules:
- Do NOT use the patient's name (use "Dear Member" or "Hi there")
- Do NOT include any PHI beyond what is in the profile
- Keep it under 150 words for SMS, 250 words for Email
- Be specific about their health situation based on the context
- End with a clear call to action
- Do NOT use generic boilerplate"""},
                            {"role": "user", "content": f"Generate a {msg_type} message for this patient:\n{context}"}
                        ]
                    )
                    message = resp.choices[0].message.content
                    st.markdown(f'<div class="msg-card">{message}</div>', unsafe_allow_html=True)

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Channel",      channel)
                    col_b.metric("Message Type", msg_type)
                    col_c.metric("Risk Level",   p['risk_level'])

                    with st.expander("View Patient Context Used"):
                        st.text(context)
                except Exception as e:
                    st.error(f"Generation error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    st.markdown("Comparison of 5 machine learning classifiers trained to predict patient missed care risk.")

    risk_model, encoders, feature_list, model_results, fi = load_models()

    if model_results is None:
        st.error("Model files not found. Please run models/train_models.py first.")
    else:
        # Dataset summary
        df_summary = query("SELECT COUNT(*) as total_patients, AVG(risk_score) as avg_risk, SUM(spoilage_flag) as missed_care FROM patients")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training Patients",  f"{int(df_summary['total_patients'][0]):,}")
        c2.metric("Avg Risk Score",     f"{df_summary['avg_risk'][0]:.1f}")
        c3.metric("Missed Care Cases",  f"{int(df_summary['missed_care'][0]):,}")
        c4.metric("Features Used",      f"{len(feature_list) if feature_list else 22}")

        st.markdown("---")
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

        # Color-coded results table
        mr = model_results.copy()
        mr['Accuracy'] = mr['Accuracy'].round(4)
        mr['F1-Score'] = mr['F1-Score'].round(4)
        mr['AUC-ROC']  = mr['AUC-ROC'].round(4)

        def color_metric(val):
            if val >= 0.75: return 'background-color: #14532d; color: #86efac'
            elif val >= 0.65: return 'background-color: #1a3a1a; color: #86efac'
            elif val >= 0.55: return 'background-color: #78350f; color: #fcd34d'
            else: return 'background-color: #7f1d1d; color: #fca5a5'

        styled = mr.style.applymap(color_metric, subset=['Accuracy','F1-Score','AUC-ROC'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        # Bar chart comparison
        with col1:
            st.markdown('<div class="section-header">Metric Comparison</div>', unsafe_allow_html=True)
            fig_bar = go.Figure()
            for metric, color in [('Accuracy','#3b82f6'),('F1-Score','#22c55e'),('AUC-ROC','#f59e0b')]:
                fig_bar.add_trace(go.Bar(
                    name=metric, x=mr['Model'], y=mr[metric], marker_color=color
                ))
            fig_bar.update_layout(
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e8eaf0', xaxis=dict(gridcolor='#2a2f3e', tickangle=-15),
                yaxis=dict(gridcolor='#2a2f3e', range=[0.5, 1.0]),
                legend=dict(font=dict(color='#e8eaf0')),
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Feature importance
        with col2:
            st.markdown('<div class="section-header">Feature Importance (Best Model)</div>', unsafe_allow_html=True)
            if fi is not None:
                fi_top = fi.head(15).sort_values('importance')
                fig_fi = px.bar(
                    fi_top, x='importance', y='feature',
                    orientation='h', color='importance',
                    color_continuous_scale='Blues'
                )
                fig_fi.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e8eaf0', xaxis=dict(gridcolor='#2a2f3e'),
                    yaxis=dict(gridcolor='#2a2f3e'), coloraxis_showscale=False,
                    margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")

        st.markdown("---")
        st.markdown('<div class="section-header">Training Data Summary</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)

        with col3:
            df_rl = query("SELECT risk_level, COUNT(*) as n FROM patients GROUP BY risk_level")
            fig_rl = px.pie(
                df_rl, names='risk_level', values='n',
                color='risk_level',
                color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'},
                hole=0.5, title="Risk Level Distribution in Training Data"
            )
            fig_rl.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font_color='#e8eaf0',
                legend=dict(font=dict(color='#e8eaf0')), margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig_rl, use_container_width=True)

        with col4:
            df_mc = query("SELECT spoilage_flag as missed_care, COUNT(*) as n FROM patients GROUP BY spoilage_flag")
            df_mc['label'] = df_mc['missed_care'].apply(lambda x: 'Missed Care' if x else 'On Track')
            fig_mc = px.pie(
                df_mc, names='label', values='n',
                color='label',
                color_discrete_map={'Missed Care':'#ef4444','On Track':'#22c55e'},
                hole=0.5, title="Target Variable Distribution"
            )
            fig_mc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font_color='#e8eaf0',
                legend=dict(font=dict(color='#e8eaf0')), margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig_mc, use_container_width=True)
