import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

BASE = Path(__file__).resolve().parent
THRESHOLD = 0.30

st.set_page_config(page_title="ChurnIQ", page_icon="📊", layout="wide")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(BASE / "churn_preprocessor.joblib")
    model = xgb.XGBClassifier()
    model.load_model(str(BASE / "churn_xgboost_native.json"))
    metadata_path = BASE / "churn_model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return preprocessor, model, float(metadata.get("threshold", THRESHOLD))

preprocessor, model, threshold = load_artifacts()

st.title("ChurnIQ")
st.caption("Customer Retention Intelligence | Machine Learning Demo")
st.write(
    "Estimate whether a telecom customer may leave. "
    "Enter the customer's details below and run the prediction."
)

with st.sidebar:
    st.header("About this model")
    st.metric("Decision threshold", f"{threshold:.0%}")
    st.write("Model: XGBoost")
    st.write("Output: estimated churn probability")
    st.warning(
        "This is a demonstration model. Predictions are estimates, "
        "not guarantees about individual customers."
    )

st.subheader("Customer profile")

with st.form("customer_form"):
    a, b, c = st.columns(3)

    with a:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Has partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])

    with b:
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

    with c:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=1000.0, value=70.0, step=1.0)
        total = st.number_input("Total charges ($)", min_value=0.0, max_value=100000.0, value=840.0, step=10.0)

    submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    try:
        transformed = preprocessor.transform(customer)
        probability = float(model.predict_proba(transformed)[0, 1])
        st.divider()
        st.subheader("Prediction result")
        left, right = st.columns(2)
        left.metric("Estimated churn probability", f"{probability:.1%}")
        right.metric("Model decision threshold", f"{threshold:.0%}")
        st.progress(min(max(probability, 0.0), 1.0))

        if probability >= threshold:
            st.error("Likely to churn — consider reviewing this customer's retention needs.")
        else:
            st.success("Likely to stay — the estimated risk is below the decision threshold.")

        st.caption(
            "The 30% threshold is the project's selected classification threshold. "
            "Probability is an estimate and should be interpreted with other evidence."
        )
    except Exception as exc:
        st.error("Prediction failed. Check that the saved preprocessor and model match.")
        st.exception(exc)

st.divider()
st.caption("Portfolio project • Customer churn prediction • Built with Python, XGBoost and Streamlit")
