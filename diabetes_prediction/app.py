# app.py - Improved Streamlit UI for Diabetes Logistic Regression
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -------------------------
# UI / app configuration
# -------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Header and short description for context
st.title("🩺 Diabetes Risk Predictor")
st.write(
    "Enter a few clinical measurements on the left and click **Predict**. "
    "The model returns the probability of diabetes and a clear suggestion."
)

# -------------------------
# Load model and scaler (cached for speed)
# -------------------------
@st.cache_resource
def load_artifacts(model_path="diabetes_prediction/diabetes_logistic_model.pkl", scaler_path="diabetes_prediction/diabetes_scaler.pkl"):
    """Load saved model and scaler once to reuse across app interactions."""
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model or scaler file not found. Make sure 'diabetes_logistic_model.pkl' "
        "and 'diabetes_scaler.pkl' are present in the same folder as this app."
    )
    st.stop()

# -------------------------
# Sidebar: collect minimal inputs
# -------------------------
st.sidebar.header("Patient measurements")
st.sidebar.markdown("Provide the values below. Defaults are typical reference values.")

# Only ask for the features used by the model (minimal required)
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
glucose = st.sidebar.slider("Glucose (mg/dL)", min_value=40, max_value=250, value=120)
bloodpressure = st.sidebar.slider("Blood Pressure (mm Hg)", min_value=30, max_value=140, value=70)
skinthickness = st.sidebar.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.sidebar.slider("Insulin (mu U/ml)", min_value=0, max_value=900, value=79)
bmi = st.sidebar.slider("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
dpf = st.sidebar.slider("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.371, step=0.001)
age = st.sidebar.slider("Age (years)", min_value=10, max_value=100, value=29)

# Optional: let user choose classification threshold
threshold = st.sidebar.slider("Probability threshold (to flag risk)", min_value=0.1, max_value=0.9, value=0.5, step=0.01)

# Button to trigger prediction
predict_btn = st.sidebar.button("Predict ✅")

# Prepare input DataFrame (order matches training features)
input_df = pd.DataFrame(
    {
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [bloodpressure],
        "SkinThickness": [skinthickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age],
    }
)
st.markdown(
        "### Quick tips\n"
        "- **Glucose** and **BMI** are often strong predictors.\n"
        "- Use the threshold slider to adjust sensitivity vs specificity.\n"
    )

# -------------------------
# Prediction flow (when button pressed)
# -------------------------
if predict_btn:
    # 1) Scale inputs to the same space the model was trained on
    scaled = scaler.transform(input_df)

    # 2) Model prediction: probability and class using the chosen threshold
    prob = model.predict_proba(scaled)[0, 1]
    pred_class = int(prob >= threshold)

    # 3) Visual feedback: big metrics and colored indicator
    st.markdown("---")
    st.subheader("Prediction Summary")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    # Show probability as percentage and interpreted status
    metric_col1.metric("Diabetes Probability", f"{prob*100:.1f} %", delta=None)
    # Show predicted class (Yes/No) with color-coded text
    if pred_class == 1:
        metric_col2.markdown("### ⚠️ Risk: **High**")
    else:
        metric_col2.markdown("### ✅ Risk: **Low**")
    metric_col3.metric("Threshold", f"{threshold:.2f}")

    # 4) Probability donut chart for visual effect and clarity
    fig, ax = plt.subplots(figsize=(4, 3))
    sizes = [prob, 1 - prob]
    colors = ["#d62728", "#1f77b4"]  # red for positive, blue for negative
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.45))
    ax.set(aspect="equal")
    plt.legend(wedges, [f"Diabetes: {prob*100:.1f}%", f"No Diabetes: {(1-prob)*100:.1f}%"], loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("Predicted Probability")
    st.pyplot(fig)

    # 5) Confidence bar - progress style to reinforce probability visually
    st.progress(int(prob * 100))

    # 6) Short actionable advice based on probability band
    st.markdown("### What this means")
    if prob >= 0.75:
        st.warning(
            "High estimated probability. Recommend urgent clinical follow-up and additional testing (OGTT/HbA1c)."
        )
    elif 0.5 <= prob < 0.75:
        st.info(
            "Moderate probability. Consider lifestyle review and further diagnostic testing as appropriate."
        )
    else:
        st.success(
            "Low estimated probability. Maintain healthy habits and routine screening."
        )

    # 8) Fun finishing touch: celebrate low-risk results
    if prob < 0.10:
        st.balloons()

# -------------------------
# Footer: notes & disclaimers
# -------------------------
st.markdown("---")
st.caption(
    "This app uses a pre-trained logistic regression model for demonstration only. "
    "It is NOT a medical diagnostic tool. For medical decisions, consult a clinician."
)
