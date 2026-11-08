"""
Credit Card Fraud Detection - Streamlit App
--------------------------------------------
This app loads the Random Forest model we trained and saved in the Jupyter
notebook, and lets the user check whether a transaction looks like fraud.

Since this dataset has 28 anonymized PCA columns (V1-V28) that nobody can
realistically fill in by hand, the app offers THREE ways to give input:

1. Upload a CSV file with transaction(s) and predict on all rows
2. A simple form where you only enter Amount + Time (the V1-V28 values are
   filled in with their average/typical values automatically)
3. Load a random real sample transaction (genuine or fraud) with one click
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------------
# Load the saved model, scaler, column order and sample data
# ---------------------------------------------------------
model = pickle.load(open("models/fraud_model.pkl", "rb"))
amount_scaler = pickle.load(open("models/amount_scaler.pkl", "rb"))
time_scaler = pickle.load(open("models/time_scaler.pkl", "rb"))
columns = pickle.load(open("models/columns.pkl", "rb"))
samples = pd.read_csv("models/sample_transactions.csv")

# these are the "typical" values for V1-V28 (their average from training data)
# we use these to auto-fill the simple form, since nobody types 28 PCA values by hand
v_columns = [c for c in columns if c.startswith("V")]
default_v_values = {col: 0.0 for col in v_columns}  # PCA components average to ~0

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.write(
    "This app predicts whether a credit card transaction is **genuine** or **fraudulent**, "
    "using a Random Forest model trained on real (anonymized) transaction data."
)

st.info(
    "Note: most columns in this dataset (V1-V28) are anonymized using PCA for privacy, "
    "so this app gives you 3 ways to test predictions - pick the tab that suits you."
)

tab1, tab2, tab3 = st.tabs(["📁 Upload CSV", "✍️ Simple Form", "🎲 Load Sample Transaction"])

# ---------------------------------------------------------
# helper function: takes a dataframe with the raw columns
# (Time, Amount, V1..V28) and returns fraud predictions
# ---------------------------------------------------------
def predict_transactions(input_df):
    df = input_df.copy()

    # scale Time and Amount using the SAME scalers used during training
    df["scaled_time"] = time_scaler.transform(df[["Time"]])
    df["scaled_amount"] = amount_scaler.transform(df[["Amount"]])

    # make sure columns are in the exact order the model expects
    df = df[columns]

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]  # probability of class 1 (fraud)

    return predictions, probabilities


# ===========================================================
# TAB 1: Upload CSV
# ===========================================================
with tab1:
    st.subheader("Upload a CSV file")
    st.write(
        "The file should have these columns: `Time`, `V1` to `V28`, `Amount` "
        "(same format as the original dataset)."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(user_df.head())

        if st.button("Predict on uploaded file"):
            try:
                preds, probs = predict_transactions(user_df)
                result_df = user_df.copy()
                result_df["Prediction"] = np.where(preds == 1, "Fraud", "Genuine")
                result_df["Fraud Probability"] = probs.round(4)

                st.write("### Results")
                st.dataframe(result_df)

                fraud_count = (preds == 1).sum()
                st.warning(f"⚠️ {fraud_count} out of {len(preds)} transaction(s) flagged as fraud.")
            except Exception as e:
                st.error(f"Something went wrong: {e}. Make sure your CSV has the right columns.")


# ===========================================================
# TAB 2: Simple Form (Amount + Time only)
# ===========================================================
with tab2:
    st.subheader("Enter basic transaction details")
    st.write(
        "We only ask for **Amount** and **Time** here, since the other 28 columns (V1-V28) "
        "are anonymized PCA values that can't be entered manually. "
        "They are automatically filled in with their typical (average) value."
    )

    amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=30000.0, value=100.0)
    time_val = st.number_input(
        "Time (seconds since the first transaction in the dataset)",
        min_value=0.0, max_value=172792.0, value=50000.0
    )

    if st.button("Predict (Simple Form)"):
        row = {"Time": time_val, "Amount": amount}
        row.update(default_v_values)
        input_df = pd.DataFrame([row])

        preds, probs = predict_transactions(input_df)
        prediction = preds[0]
        probability = probs[0]

        st.markdown("### Result")
        if prediction == 1:
            st.error(f"⚠️ This transaction looks **FRAUDULENT**.\n\nFraud probability: **{probability*100:.2f}%**")
        else:
            st.success(f"✅ This transaction looks **GENUINE**.\n\nFraud probability: **{probability*100:.2f}%**")

        st.progress(float(probability))
        st.caption(
            "Note: since V1-V28 are set to average values here, this is a simplified "
            "prediction. Real fraud detection relies heavily on those PCA features too."
        )


# ===========================================================
# TAB 3: Load a random real sample transaction
# ===========================================================
with tab3:
    st.subheader("Try a real sample transaction")
    st.write(
        "These are real transactions taken from the test set (not used during training), "
        "so we already know whether they were genuine or fraud. This is a good way to "
        "sanity-check that the model works correctly."
    )

    sample_type = st.radio("Pick a sample type:", ["Random Genuine transaction", "Random Fraud transaction"])

    if st.button("Load Sample"):
        if sample_type == "Random Genuine transaction":
            row = samples[samples["Class"] == 0].sample(1)
        else:
            row = samples[samples["Class"] == 1].sample(1)

        st.session_state["sample_row"] = row

    if "sample_row" in st.session_state:
        row = st.session_state["sample_row"]
        actual_label = "Fraud" if row["Class"].values[0] == 1 else "Genuine"

        st.write("Selected transaction (scaled values, as stored after preprocessing):")
        st.dataframe(row)

        if st.button("Predict on this sample"):
            input_df = row[columns]  # already scaled + in the right column order
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.markdown("### Result")
            if prediction == 1:
                st.error(f"⚠️ Model predicts: **FRAUD** (probability: {probability*100:.2f}%)")
            else:
                st.success(f"✅ Model predicts: **GENUINE** (probability: {probability*100:.2f}%)")

            st.caption(f"Actual label for this transaction: **{actual_label}**")


st.markdown("---")
st.caption(
    "This app uses a Random Forest model trained on the Kaggle Credit Card Fraud Detection dataset. "
    "Built as part of a college ML/DL capstone project."
)
