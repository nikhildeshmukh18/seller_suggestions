import streamlit as st
import pandas as pd
from utils import call_groq_llm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Seller Assistant", layout="centered")
st.title("🛍️ Seller AI Assistant (MVP)")

# Tabs
tabs = st.tabs(["📝 Content Generator", "🛒 Suggestions", "🧠 Sentiment", "📦 Forecasting"])

# --- 1. Content Generation ---
with tabs[0]:
    st.subheader("Generate Product Description")
    title = st.text_input("Product Title")
    features = st.text_area("Key Features (optional)")
    if st.button("Generate Description"):
        prompt = f"Write a compelling e-commerce product description for:\nTitle: {title}\nFeatures: {features}"
        st.success(call_groq_llm(prompt))

# --- 2. Custom Suggestions ---
with tabs[1]:
    st.subheader("Get Product Suggestions")
    query = st.text_input("Current Product")
    if st.button("Suggest Similar Products"):
        prompt = f"Suggest 3 upsell or cross-sell products for: {query}"
        st.info(call_groq_llm(prompt))

# --- 3. Sentiment Analysis ---
with tabs[2]:
    st.subheader("Analyze Customer Reviews")
    review = st.text_area("Paste customer reviews (separated by line)")
    if st.button("Analyze Sentiment"):
        prompt = f"Analyze these reviews for sentiment and suggest improvements:\n{review}"
        st.warning(call_groq_llm(prompt))

# --- 4. Inventory Forecasting ---
with tabs[3]:
    st.subheader("Simple Demand Forecasting")
    sample = pd.DataFrame({
        "month": [1, 2, 3, 4, 5, 6],
        "units_sold": [120, 150, 160, 180, 170, 200]
    })
    st.dataframe(sample)
    if st.button("Forecast Next Month"):
        X = sample[["month"]]
        y = sample["units_sold"]
        model = LinearRegression().fit(X, y)
        next_month = model.predict([[7]])[0]
        st.success(f"📦 Forecast for month 7: {int(next_month)} units")

        # Plot
        plt.plot(sample["month"], sample["units_sold"], marker='o', label='Historical')
        plt.plot(7, next_month, 'ro', label='Forecast')
        plt.xlabel("Month")
        plt.ylabel("Units Sold")
        plt.legend()
        st.pyplot(plt.gcf())
