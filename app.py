import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="🩺 COVID Health Tracker", layout="wide", page_icon="💉")
st.title("🩺 COVID / Health Tracker Dashboard")

# Sidebar - Upload
st.sidebar.header("Upload Your CSV")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with Text, Symptoms, and Severity columns",
    type=["csv"]
)

# Functions
def sentiment_analysis(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def risk_level(severity):
    if severity >= 4:
        return "High"
    elif severity >= 2:
        return "Medium"
    else:
        return "Low"

# Main
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Sentiment & Risk
    df['Sentiment'] = df['Text'].apply(sentiment_analysis)
    df['Risk'] = df['Severity'].apply(risk_level)
    
    # Symptom Classification
    symptom_categories = {
        "Respiratory":["cough","breathing","shortness","respiratory"],
        "Fever":["fever","temperature"],
        "Fatigue":["tired","fatigue","weakness"],
        "Headache":["headache","migraine"]
    }
    
    def classify_symptom(text):
        text = str(text).lower()
        categories = []
        for key, keywords in symptom_categories.items():
            if any(word in text for word in keywords):
                categories.append(key)
        if categories:
            return ", ".join(categories)
        else:
            return "Other"
    
    df['Symptom_Category'] = df['Text'].apply(classify_symptom)
    
    # Summary Cards
    st.subheader("📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reports", len(df))
    col2.metric("High Risk ⚠️", len(df[df['Risk']=="High"]))
    col3.metric("Medium Risk ⚠️", len(df[df['Risk']=="Medium"]))
    col4.metric("Low Risk ✅", len(df[df['Risk']=="Low"]))
    
    # Sentiment Pie Chart
    st.subheader("Sentiment Distribution")
    fig_sent = px.pie(df, names="Sentiment", color="Sentiment",
                      color_discrete_map={"Positive":"green","Neutral":"gray","Negative":"red"},
                      hole=0.4)
    st.plotly_chart(fig_sent, use_container_width=True)
    
    # Risk Heatmap - Symptom vs Risk
    st.subheader("Risk Heatmap by Symptom Category")
    heat_data = pd.crosstab(df['Symptom_Category'], df['Risk'])
    fig, ax = plt.subplots()
    sns.heatmap(heat_data, annot=True, cmap="Reds", fmt="d", ax=ax)
    st.pyplot(fig)
    
    # Trend Lines (Optional: if CSV has Date column)
    if 'Date' in df.columns:
        st.subheader("Symptom Frequency Over Time")
        df['Date'] = pd.to_datetime(df['Date'])
        trend = df.groupby(['Date','Symptom_Category']).size().reset_index(name='Count')
        fig_trend = px.line(trend, x='Date', y='Count', color='Symptom_Category')
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Top Reports Table
    st.subheader("Top High Risk Reports")
    st.dataframe(df[df['Risk']=="High"][['Text','Symptoms','Severity','Sentiment','Symptom_Category']].head(10))
    
else:
    st.info("Upload a CSV file with Text, Symptoms, and Severity columns to analyze health data.")
