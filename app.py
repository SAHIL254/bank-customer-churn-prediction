import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib as mpl

# ---------------------------
# Matplotlib Safety (CRITICAL)
# ---------------------------
mpl.rcParams["text.usetex"] = False
plt.rcParams.update({
    "figure.dpi": 100,
    "figure.autolayout": False
})

# ---------------------------
# Config
# ---------------------------
THRESHOLD = 0.45   # Chosen to prioritize churn recall over accuracy

st.set_page_config(
    page_title="Bank Churn Predictor",
    layout="wide"
)

# ---------------------------
# Load Model (Cached)
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("artifacts/churn_model.pkl")

model = load_model()

# ---------------------------
# Background Data for SHAP
# ---------------------------
def load_background_data():
    df = pd.read_csv("DATA/Churn_Modelling.csv")
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
    return df.sample(100, random_state=42)

X_background_raw = load_background_data()
X_background = model.named_steps["preprocessing"].transform(X_background_raw)

# ✅ Get transformed feature names ONCE
feature_names = model.named_steps["preprocessing"].get_feature_names_out()

# ---------------------------
# SHAP Explainer (FIXED)
# ---------------------------
@st.cache_resource
def load_shap_explainer(_model, X_background, feature_names):
    return shap.LinearExplainer(
        _model.named_steps["classifier"],
        X_background,
        feature_names=feature_names
    )

explainer = load_shap_explainer(model, X_background, feature_names)

# ---------------------------
# Styling
# ---------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(45deg, #1e90ff, #ff6347); color: white; }
.result-card { background-color: rgba(0, 0, 0, 0.5); padding: 20px; border-radius: 15px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.title("📋 Customer Information")

credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 80, 40)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance", min_value=0.0, step=1000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])

has_card = st.sidebar.selectbox("Has Credit Card?", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member?", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, step=5000.0)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

st.sidebar.markdown("---")
st.sidebar.caption("Results update automatically as you change inputs.")

# ---------------------------
# Title
# ---------------------------
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("""
This application predicts whether a bank customer is likely to **churn**.
Decision threshold = **0.45**.
""")

st.divider()

# ---------------------------
# Prepare Input
# ---------------------------
input_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary,
    "Geography": geography,
    "Gender": gender
}])

churn_prob = model.predict_proba(input_data)[0][1]
retention_prob = 1 - churn_prob
prediction = int(churn_prob >= THRESHOLD)

# ---------------------------
# Prediction Result
# ---------------------------
st.subheader("🎯 Prediction Result")

if prediction == 1:
    st.error(f"⚠️ High Churn Risk ({churn_prob:.2%})")
    st.markdown("""
    **Recommended Actions**
    - Proactively contact customer  
    - Offer retention incentives  
    - Improve engagement
    """)
else:
    st.success(f"✅ Low Churn Risk ({retention_prob:.2%})")
    st.markdown("""
    **Suggested Actions**
    - Maintain engagement  
    - Monitor customer behavior
    """)

st.caption(f"Decision Threshold Used: **{THRESHOLD}**")
st.caption(
    "Threshold chosen to prioritize churn recall over accuracy."
)


# ---------------------------
# Top row: Gauge + SHAP
# ---------------------------
col1, col2 = st.columns([1, 2])

# ---------------------------
# Gauge
# ---------------------------
with col1:
    st.subheader("📊 Churn Risk Gauge")

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        number={'suffix': "%"},
        title={'text': "Churn Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 40], 'color': "#2ecc71"},
                {'range': [40, 65], 'color': "#f1c40f"},
                {'range': [65, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': THRESHOLD * 100
            }
        }
    ))

    gauge_fig.update_layout(height=350, margin=dict(l=30, r=30, t=60, b=20))
    st.plotly_chart(gauge_fig, width="stretch")

# ---------------------------
# SHAP Explanation (FIXED)
# ---------------------------
with col2:
    st.subheader("🧠 SHAP Explanation (Top Drivers)")

    X_input_transformed = model.named_steps["preprocessing"].transform(input_data)
    shap_values = explainer(X_input_transformed)

    fig = plt.figure(figsize=(8, 4))
    shap.plots.waterfall(
        shap_values[0],
        max_display=8,
        show=False
    )

    st.pyplot(fig, width="content")
    plt.close(fig)

# ---------------------------
# Pie Chart
# ---------------------------
st.subheader("📈 Risk Distribution")

labels = ["Low Risk (Retained)", "High Risk (Churn)"]
sizes = [retention_prob * 100, churn_prob * 100]
colors = ["#2ecc71", "#e74c3c"]

fig, ax = plt.subplots(figsize=(6, 5), facecolor="#f0f0f0")

wedges, texts, autotexts = ax.pie(
    sizes,
    autopct="%1.1f%%",
    startangle=90,
    explode=(0.05, 0.05),
    colors=colors,
    textprops={"fontsize": 14, "weight": "bold", "color": "white"}
)

ax.legend(
    wedges,
    labels,
    title="Risk Category",
    loc="upper left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)

ax.set_title("Churn Risk Distribution", fontsize=14, weight="bold")
ax.axis("equal")

st.pyplot(fig, width="content")
plt.close(fig)


