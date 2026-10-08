"""Customer Retention Intelligence — bank churn risk assessment.

Loads the pre-trained artifacts from "Bank Churn.ipynb":
  - random_forest_model.pkl (RandomForestClassifier, 200 trees, max_depth=10)
  - churn_scaler.pkl        (StandardScaler fit on the 11 training columns)

Pipeline (exactly as trained — never retrained here):
  User Input -> Feature Encoding -> DataFrame in exact feature order
  -> Saved Scaler -> Saved Random Forest -> Churn Probability -> Risk Assessment

Run with:
    streamlit run bank_churn_app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "churn_scaler.pkl"

# Exact training column order (Bank Churn.ipynb, cells 10-12).
FEATURES = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
]

# Gender encoding used in training: Male -> 1, Female -> 0.
GENDER_MAP = {"Male": 1, "Female": 0}
GENDER_ICONS = {"Male": "♂", "Female": "♀"}

# Geography one-hot encoding used in training (France is the reference).
GEOGRAPHY_MAP = {
    "France": (0, 0),
    "Germany": (1, 0),
    "Spain": (0, 1),
}

# Flag presentation (labels only — model still gets the codes above).
GEOGRAPHY_FLAGS = {"France": "🇫🇷", "Germany": "🇩🇪", "Spain": "🇪🇸"}

# National-color ribbons: France blue-white-red, Germany black-red-gold,
# Spain red-gold-red.
COUNTRY_RIBBON = {
    "France": "linear-gradient(90deg, #0055A4 0 33%, #FFFFFF 33% 66%, #EF4135 66% 100%)",
    "Germany": "linear-gradient(90deg, #000000 0 33%, #DD0000 33% 66%, #FFCE00 66% 100%)",
    "Spain": "linear-gradient(90deg, #AA151B 0 25%, #F1BF00 25% 75%, #AA151B 75% 100%)",
}


def country_ribbon_html(geography):
    """Tricolor ribbon in the customer's national colors."""
    return (
        f'<div style="height:10px;border-radius:999px;margin:0.4rem 0 0.2rem;'
        f"background:{COUNTRY_RIBBON[geography]};"
        'box-shadow:0 2px 8px rgba(0,0,0,0.2);"></div>'
        f'<div style="font-weight:700;">{GEOGRAPHY_FLAGS[geography]} {geography}</div>'
    )

st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="🏛️",
    layout="wide",
)


# ------------------------------------------------------------------
# Model pipeline
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the saved Random Forest and scaler once; raises FileNotFoundError."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def prepare_input(credit_score, gender, age, tenure, balance, num_products,
                  has_card, is_active, salary, geography):
    """Encode one customer's raw inputs into the exact 11-column model frame."""
    geo_germany, geo_spain = GEOGRAPHY_MAP[geography]
    row = {
        "CreditScore": int(credit_score),
        "Gender": GENDER_MAP[gender],
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": int(has_card),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": float(salary),
        "Geography_Germany": geo_germany,
        "Geography_Spain": geo_spain,
    }
    return pd.DataFrame([{c: row[c] for c in FEATURES}])


def predict_churn(model, scaler, input_df):
    """Scale with the saved scaler and return (label, churn probability)."""
    input_scaled = scaler.transform(input_df)
    prediction = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0, 1])
    return prediction, probability


def risk_band(probability):
    """Map churn probability to a Low / Medium / High risk band."""
    if probability < 0.30:
        return "Low Risk"
    if probability <= 0.60:
        return "Medium Risk"
    return "High Risk"


# Sophisticated banking theme: deep-navy hero, gold accents,
# soft colorful cards. Pipeline functions above are untouched.
THEME_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
.hero {
    background: linear-gradient(120deg, #0B1F3A 0%, #134E5E 60%, #0E7C6B 100%);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    color: #FFFFFF;
    box-shadow: 0 12px 32px rgba(11, 31, 58, 0.35);
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #FFFFFF !important;
    margin: 0 0 0.3rem !important;
    font-size: 2.4rem !important;
}
.hero p { color: #D9E8E3 !important; margin: 0; font-size: 1.05rem; }
.hero .gold-rule {
    width: 72px; height: 4px; border-radius: 4px;
    background: linear-gradient(90deg, #D4AF37, #F6E27A);
    margin-bottom: 0.9rem;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #B8860B, #D4AF37);
    border: none; color: #0B1F3A !important;
    font-weight: 800; font-size: 1.05rem;
    border-radius: 12px; padding: 0.7rem 1.4rem; width: 100%;
    box-shadow: 0 6px 18px rgba(212, 175, 55, 0.4);
}
div.stButton > button[kind="primary"]:hover {
    filter: brightness(1.08);
}
.risk-card {
    border-radius: 16px; padding: 1.6rem 1.8rem; color: #FFFFFF;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.25);
    text-align: center; margin-bottom: 1rem;
}
.risk-low { background: linear-gradient(120deg, #0E7C6B, #34D399); }
.risk-medium { background: linear-gradient(120deg, #B45309, #F59E0B); }
.risk-high { background: linear-gradient(120deg, #7F1D1D, #EF4444); }
.risk-card .status { font-size: 1.9rem; font-weight: 800; margin: 0; }
.risk-card .prob { font-size: 3rem; font-weight: 800; margin: 0.1rem 0; }
.risk-card .band {
    display: inline-block; background: rgba(255, 255, 255, 0.25);
    border-radius: 999px; padding: 0.25rem 1.1rem; font-weight: 700;
}
.section-card {
    border-radius: 16px; padding: 0.4rem 0.2rem;
}
.stMetric {
    background: linear-gradient(135deg, #EFF6FF, #E0F2FE);
    border: 1px solid #BAE6FD; border-radius: 14px; padding: 0.8rem;
}
</style>
"""


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
def main():
    try:
        model, scaler = load_model()
    except FileNotFoundError as err:
        st.error(
            "Model files not found. Make sure 'random_forest_model.pkl' and "
            f"'churn_scaler.pkl' are in the same folder as this script. "
            f"(Details: {err})"
        )
        st.stop()
    except Exception:
        st.error("Could not load the model files. Check they are valid and try again.")
        st.stop()

    # ---- Header ----
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><div class="gold-rule"></div>'
        '<div style="display:flex;align-items:center;gap:0.9rem;">'
        '<svg width="52" height="52" viewBox="0 0 64 64" fill="none">'
        "<defs><linearGradient id='gold' x1='0' y1='0' x2='1' y2='1'>"
        "<stop offset='0' stop-color='#F6E27A'/>"
        "<stop offset='1' stop-color='#D4AF37'/></linearGradient></defs>"
        "<path d='M32 6 6 24h52L32 6z' fill='url(#gold)'/>"
        "<rect x='10' y='28' width='6' height='20' rx='1.5' fill='url(#gold)'/>"
        "<rect x='21' y='28' width='6' height='20' rx='1.5' fill='url(#gold)'/>"
        "<rect x='29' y='28' width='6' height='20' rx='1.5' fill='url(#gold)'/>"
        "<rect x='37' y='28' width='6' height='20' rx='1.5' fill='url(#gold)'/>"
        "<rect x='48' y='28' width='6' height='20' rx='1.5' fill='url(#gold)'/>"
        "<rect x='6' y='50' width='52' height='6' rx='2' fill='url(#gold)'/>"
        "</svg>"
        "<div><h1 style='margin:0 !important;'>Customer Retention Intelligence</h1>"
        "<p>AI-powered customer churn risk assessment</p></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Appearance switch: light (white) <-> dark (black) dashboard.
    dark_mode = st.toggle("🌙 Dark background", key="dark_mode")
    if dark_mode:
        st.markdown(
            """
            <style>
            [data-testid="stAppViewContainer"] { background: #0B0E14; }
            [data-testid="stAppViewContainer"] p,
            [data-testid="stAppViewContainer"] li,
            [data-testid="stMarkdownContainer"] p { color: #E2E8F0; }
            h1, h2, h3 { color: #F8FAFC !important; }
            [data-testid="stWidgetLabel"] p { color: #E2E8F0 !important; }
            [data-testid="stCaptionContainer"] { color: #94A3B8 !important; }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
            }
            .stMetric {
                background: rgba(255, 255, 255, 0.05) !important;
                border: 1px solid rgba(255, 255, 255, 0.14) !important;
            }
            .stMetric label, .stMetric [data-testid="stMetricValue"] {
                color: #F8FAFC !important;
            }
            [data-testid="stExpander"] {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.12);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # ---- Input sections ----
    with st.container(border=True):
        st.subheader("Customer Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.number_input("Credit Score", 300, 850, 650, 1)
            gender = st.selectbox(
                "Gender", ["Male", "Female"],
                format_func=lambda c: f"{GENDER_ICONS[c]} {c}",
            )
        with c2:
            age = st.number_input("Age", 18, 100, 38, 1)
            geography = st.selectbox(
                "Geography", ["France", "Germany", "Spain"],
                format_func=lambda c: f"{GEOGRAPHY_FLAGS[c]} {c}",
            )
            st.markdown(country_ribbon_html(geography), unsafe_allow_html=True)
        with c3:
            tenure = st.number_input("Tenure (years with bank)", 0, 10, 5, 1)

    with st.container(border=True):
        st.subheader("Financial Information")
        f1, f2 = st.columns(2)
        with f1:
            balance = st.number_input("Balance ($)", 0.0, 300000.0, 60000.0, 1000.0)
        with f2:
            salary = st.number_input(
                "Estimated Salary ($)", 0.0, 300000.0, 100000.0, 1000.0)

    with st.container(border=True):
        st.subheader("Banking Activity")
        b1, b2, b3 = st.columns(3)
        with b1:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
        with b2:
            has_card = int(st.selectbox("Has Credit Card", ["Yes", "No"]) == "Yes")
        with b3:
            is_active = st.selectbox("Is Active Member", ["Yes", "No"]) == "Yes"
            is_active = int(is_active)

    assess = st.button("Assess Churn Risk", type="primary")

    # ---- Prediction section ----
    if assess:
        try:
            input_df = prepare_input(
                credit_score, gender, age, tenure, balance,
                num_products, has_card, is_active, salary, geography,
            )
            prediction, probability = predict_churn(model, scaler, input_df)
        except Exception:
            st.error("Could not compute the assessment. "
                     "Check the inputs and try again.")
        else:
            st.divider()
            st.subheader("Customer Risk Assessment")
            status = "Likely to Churn" if prediction == 1 else "Likely to Stay"
            band = risk_band(probability)
            dot = {"Low Risk": "🟢", "Medium Risk": "🟡",
                   "High Risk": "🔴"}[band]
            st.markdown(
                f"**{dot} {status}**, churn probability "
                f"**{probability * 100:.1f}%** · {band}"
            )
            badge = (
                '<span style="display:inline-block;width:12px;height:12px;'
                "border-radius:3px;background:{color};margin-right:0.45rem;"
                'vertical-align:baseline;"></span>'
            )
            small = "font-size:0.85rem;color:#4B5563;"
            st.markdown(
                "**What the risk levels mean:**\n\n"
                f"<span style='{small}'>"
                f"{badge.format(color='#22C55E')}"
                f"{'**' if band == 'Low Risk' else ''}"
                "Low Risk (below 30%), comfortably safe, routine monitoring"
                f"{'**' if band == 'Low Risk' else ''}</span>\n\n"
                f"<span style='{small}'>"
                f"{badge.format(color='#F59E0B')}"
                f"{'**' if band == 'Medium Risk' else ''}"
                "Medium Risk (30% to 60%), uncertain, worth a preventive check-in"
                f"{'**' if band == 'Medium Risk' else ''}</span>\n\n"
                f"<span style='{small}'>"
                f"{badge.format(color='#EF4444')}"
                f"{'**' if band == 'High Risk' else ''}"
                "High Risk (above 60%), act now to retain this customer"
                f"{'**' if band == 'High Risk' else ''}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                country_ribbon_html(geography),
                unsafe_allow_html=True,
            )
            if prediction == 1:
                st.warning(
                    "This customer shows characteristics associated with "
                    "higher churn risk. Consider reviewing their engagement, "
                    "product usage, and relationship with the bank."
                )
            else:
                st.success(
                    "This customer currently shows a lower estimated "
                    "churn risk."
                )

    # ---- Model section ----
    st.divider()
    with st.expander("About this assessment"):
        st.markdown(
            "**Model**\n\nRandom Forest Classifier\n\n"
            "**Purpose**\n\nPredict customer churn probability"
        )


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
# pip install streamlit pandas numpy joblib scikit-learn
# streamlit run bank_churn_app.py
