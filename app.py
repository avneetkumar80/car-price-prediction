"""
Indian Pre-Owned Car Price Prediction — Streamlit Application
================================================================
Component 1B of the capstone project, rebuilt against the OFFICIAL
dataset and the model trained in project_notebook.ipynb
(model/trained_model.joblib). Serves:
  - Real-time single-car price prediction (sidebar inputs)
  - ±5% valuation banding around the point prediction
  - Batch Processing tab (upload a Cap_Test_Data_2025-shaped CSV -> download predictions)
  - Visual Insights dashboard (Age vs Price, Distance vs Price)

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Indian Pre-Owned Car Price Predictor", layout="wide")

TRAIN_DATA_PATH = "data/Cap_Training_Data_2025.csv"
MODEL_PATH = "model/trained_model.joblib"
METADATA_PATH = "model/model_metadata.joblib"

DISTANCE_CAP = 500000        # values above this are treated as data errors (see notebook Sec 3.4)
DISPLACEMENT_SENTINEL = 32000
POWER_CAP = 500


# ---------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metadata():
    return joblib.load(METADATA_PATH)


@st.cache_data
def load_training_data():
    """Load the official training data for populating dropdown choices
    and the Visual Insights dashboard, applying the same structural
    fixes as the notebook (column-name whitespace, sentinel/implausible
    value correction) so charts/dropdowns reflect what the model
    actually trained on."""
    df = pd.read_csv(TRAIN_DATA_PATH)
    df.columns = df.columns.str.strip()
    df.loc[df["engine_displacement"] == DISPLACEMENT_SENTINEL, "engine_displacement"] = np.nan
    df.loc[df["engine_power"] > POWER_CAP, "engine_power"] = np.nan
    df.loc[df["Distance"] > DISTANCE_CAP, "Distance"] = np.nan
    df = df[df["Price"] >= 1000]  # drop known target data-entry errors
    return df


with st.spinner("Loading trained model and official dataset..."):
    model_pipe = load_model()
    metadata = load_metadata()
    train_df = load_training_data()

NUMERIC_FEATURES = metadata["numeric_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]
FEATURE_COLS = metadata["feature_cols"]
BEST_MODEL_NAME = metadata["best_model_name"]

st.success("Model and dataset loaded successfully.")

st.title("🚗 Indian Pre-Owned Car Price Predictor")
st.caption(
    f"Deployed model: **{BEST_MODEL_NAME}** — trained in `project_notebook.ipynb` "
    "on the official capstone dataset (Cap_Training_Data_2025.csv)."
)

tab_predict, tab_batch, tab_insights = st.tabs(
    ["🔮 Price Prediction", "📂 Batch Processing", "📊 Visual Insights"]
)

# ---------------------------------------------------------------------
# TAB 1 — Real-time single prediction
# ---------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter Car Details")
    st.sidebar.header("Car Characteristics")

    maker_options = sorted(train_df["Maker"].unique())
    location_options = sorted(train_df["Location"].dropna().unique())
    owner_options = sorted(train_df["Owner Type"].unique())
    body_type_options = ["Unknown"] + sorted(train_df["body_type"].dropna().unique().tolist())
    transmission_options = sorted(train_df["transmission"].unique())
    fuel_type_options = sorted(train_df["fuel_type"].unique())

    maker = st.sidebar.selectbox("Maker", maker_options)
    model_choices = sorted(train_df.loc[train_df["Maker"] == maker, "model"].unique())
    car_model = st.sidebar.selectbox("Model", model_choices)
    location = st.sidebar.selectbox("Location", location_options)
    owner_type = st.sidebar.selectbox("Owner Type", owner_options)
    body_type = st.sidebar.selectbox("Body Type", body_type_options)
    transmission = st.sidebar.selectbox("Transmission", transmission_options)
    fuel_type = st.sidebar.selectbox("Fuel Type", fuel_type_options)

    st.sidebar.markdown("---")
    age = st.sidebar.slider("Age of car (years)", 0, 30, 5)
    distance = st.sidebar.slider("Distance driven (km)", 0, 500000, 45000, step=1000)
    engine_displacement = st.sidebar.slider("Engine displacement (cc)", 600, 6000, 1600, step=50)
    engine_power = st.sidebar.slider("Engine power (bhp)", 30, 500, 100, step=1)
    vroom_rating = st.sidebar.slider("Vroom Audit Rating", 4, 8, 6, step=1)
    door_count = st.sidebar.selectbox("Door Count", [2, 3, 4, 5], index=2)
    seat_count = st.sidebar.selectbox("Seat Count", [2, 4, 5, 7], index=1)

    input_row = pd.DataFrame([{
        "Maker": maker, "model": car_model, "Location": location,
        "Distance": distance, "Owner Type": owner_type, "Age of car": age,
        "engine_displacement": engine_displacement, "engine_power": engine_power,
        "body_type": None if body_type == "Unknown" else body_type,
        "Vroom Audit Rating": vroom_rating,
        "transmission": transmission, "door_count": door_count,
        "seat_count": seat_count, "fuel_type": fuel_type,
    }])[FEATURE_COLS]

    st.dataframe(input_row, use_container_width=True)

    if st.button("Predict Price", type="primary"):
        prediction = model_pipe.predict(input_row)[0]
        lower_bound = prediction * 0.95
        upper_bound = prediction * 1.05

        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Price", f"₹{prediction:,.0f}")
        col2.metric("Fair Value — Lower (-5%)", f"₹{lower_bound:,.0f}")
        col3.metric("Fair Value — Upper (+5%)", f"₹{upper_bound:,.0f}")
        st.info(
            f"**Valuation Range:** ₹{lower_bound:,.0f} – ₹{upper_bound:,.0f} "
            "(a ±5% band around the point prediction — not a statistical confidence interval)."
        )

# ---------------------------------------------------------------------
# TAB 2 — Batch processing
# ---------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Prediction")
    st.write(
        "Upload a CSV with the same feature columns as the official test "
        "dataset (`Cap_Test_Data_2025.csv`, no `Price` column) to generate "
        "bulk predictions."
    )

    uploaded_file = st.file_uploader("Upload test-shaped CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_df.columns = batch_df.columns.str.strip()

            missing_cols = [c for c in FEATURE_COLS if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Uploaded file is missing required columns: {missing_cols}")
            else:
                clean_df = batch_df.copy()
                clean_df.loc[clean_df["engine_displacement"] == DISPLACEMENT_SENTINEL, "engine_displacement"] = np.nan
                clean_df.loc[clean_df["engine_power"] > POWER_CAP, "engine_power"] = np.nan
                clean_df.loc[clean_df["Distance"] > DISTANCE_CAP, "Distance"] = np.nan

                X_batch = clean_df[FEATURE_COLS]
                preds = model_pipe.predict(X_batch)

                result_df = pd.DataFrame({
                    "ID": batch_df["ID"] if "ID" in batch_df.columns else range(1, len(batch_df) + 1),
                    "Price": np.round(preds, 2),
                })

                st.success(f"Generated predictions for {len(result_df)} rows.")
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions (CSV)",
                    data=csv_bytes,
                    file_name="Sample_Submission_Predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process the uploaded file: {e}")

# ---------------------------------------------------------------------
# TAB 3 — Visual insights
# ---------------------------------------------------------------------
with tab_insights:
    st.subheader("How Age and Distance Affect Price")
    st.write("Charts below use the official training data (post data-quality cleaning).")

    insight_df = train_df.dropna(subset=["Age of car", "Distance", "Price"]).copy()

    st.markdown("**Average Price by Age of Car**")
    age_price = insight_df.groupby("Age of car")["Price"].mean().sort_index()
    st.bar_chart(age_price)

    st.markdown("**Average Price by Distance Bucket (km driven)**")
    insight_df["distance_bucket"] = pd.cut(
        insight_df["Distance"],
        bins=[0, 20000, 40000, 60000, 80000, 100000, 200000, 500000],
        labels=["0-20k", "20-40k", "40-60k", "60-80k", "80-100k", "100-200k", "200-500k"],
    )
    dist_price = insight_df.groupby("distance_bucket", observed=True)["Price"].mean()
    st.bar_chart(dist_price)

    st.markdown("**Price vs Age of Car (scatter)**")
    st.scatter_chart(insight_df.sample(min(4000, len(insight_df)), random_state=42), x="Age of car", y="Price")

    st.markdown("**Price vs Distance Driven (scatter)**")
    st.scatter_chart(insight_df.sample(min(4000, len(insight_df)), random_state=42), x="Distance", y="Price")

    st.markdown("**Average Price by Maker**")
    maker_price = insight_df.groupby("Maker")["Price"].mean().sort_values(ascending=False)
    st.bar_chart(maker_price)