import streamlit as st
import pandas as pd
import joblib

from feature_engineer import engineer_features_and_labels, preprocess_for_sklearn

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

st.title("📈 Nifty Trading Model")
st.write("Upload OHLCV+OI data to get trading predictions.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Feature engineering (ignore labels for inference)
    X, _ = engineer_features_and_labels(df)

    # Preprocess for sklearn
    X_processed, _ = preprocess_for_sklearn(X)

    # Predictions
    preds = model.predict(X_processed)
    X["Prediction"] = preds

    # Show predictions
    st.write("### Predictions")
    st.dataframe(X[["datetime", "close", "Prediction"]])

    # Download button
    csv_out = X.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions",
        csv_out,
        "predictions.csv",
        "text/csv"
    )
