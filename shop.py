# =====================================
# Shopping Sales Prediction - Streamlit
# =====================================

import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained objects
# -------------------------------
model = joblib.load("shopping_sales_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")
model_features = joblib.load("model_features.pkl")  # MUST match training

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Shopping Sales Prediction", layout="centered")
st.title("🛒 Shopping Sales Prediction")
st.write("Predict **Purchase Amount (USD)** using customer shopping behavior")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Customer Information")

age = st.sidebar.number_input("Age", 18, 80, 30)
previous_purchases = st.sidebar.number_input("Previous Purchases", 0, 100, 5)
review_rating = st.sidebar.slider("Review Rating", 1.0, 5.0, 3.5)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
season = st.sidebar.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
category = st.sidebar.selectbox("Category", ["Clothing", "Footwear", "Accessories"])
color = st.sidebar.selectbox("Color", ["Red", "Blue", "Green", "Black"])
location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])
item_purchased = st.sidebar.selectbox(
    "Item Purchased",
    ["Shirt", "Shoes", "Bag", "Watch", "Jacket"]
)
frequency = st.sidebar.slider("Frequency of Purchases", 1, 30, 5)

# -------------------------------
# Create input dataframe EXACTLY like training
# -------------------------------
input_df = pd.DataFrame(0, index=[0], columns=model_features)

# -------------------------------
# Assign values safely
# -------------------------------
def safe_set(col, value):
    if col in input_df.columns:
        input_df[col] = value

safe_set("Age", age)
safe_set("Previous Purchases", previous_purchases)
safe_set("Review Rating", review_rating)
safe_set("Frequency of Purchases", frequency)

safe_set("Gender", 1 if gender == "Male" else 0)
safe_set("Season", ["Winter", "Spring", "Summer", "Fall"].index(season))
safe_set("Category", ["Clothing", "Footwear", "Accessories"].index(category))
safe_set("Color", ["Red", "Blue", "Green", "Black"].index(color))
safe_set("Location", ["Urban", "Suburban", "Rural"].index(location))
safe_set("Item Purchased", ["Shirt", "Shoes", "Bag", "Watch", "Jacket"].index(item_purchased))

# -------------------------------
# Ensure correct column order
# -------------------------------
input_df = input_df[model_features]

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Purchase Amount"):
    input_scaled = scaler.transform(input_df)
    input_selected = selector.transform(input_scaled)
    prediction = model.predict(input_selected)[0]

    st.success(f"💰 Predicted Purchase Amount: **${prediction:.2f}**")

    st.write("### 🔍 Input Summary")
    st.dataframe(input_df)

# -------------------------------
# Footer
# -------------------------------

