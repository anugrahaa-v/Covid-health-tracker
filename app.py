import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🩺 COVID Health Tracker", layout="wide", page_icon="💉")
st.title("🩺 COVID / Health Tracker Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Upload Your CSV")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with columns: Text, Symptoms, Severity, (optional Date)",
    type=["csv"]
)

# ---------------- FUNCTIONS ----------------
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

symptom_categories = {
    "Respiratory":["cough","breathing","shortness","respiratory"],
    "Fever":["fever","temperature"],
    "Fatigue":["tired","fatigue","weakness"],
    "Headache":["headache","migraine"]
}

def classify_symptom(text):
    text = str(text).lower()
    categories = [k for k, v in symptom_categories.items() if any(word in text for word in v)]
    return ", ".join(categories) if categories else "Other"

# ---------------- MAIN ----------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- AI/ML Features ---
    df['Sentiment'] = df['Text'].apply(sentiment_analysis)
    df['Risk'] = df['Severity'].apply(risk_level)
    df['Symptom_Category'] = df['Text'].apply(classify_symptom)
    
    # --- INTERACTIVE FILTERS ---
    risk_filter = st.sidebar.multiselect("Filter by Risk", df['Risk'].unique(), default=df['Risk'].unique())
    symptom_filter = st.sidebar.multiselect("Filter by Symptom Category", df['Symptom_Category'].unique(), default=df['Symptom_Category'].unique())
    sentiment_filter = st.sidebar.multiselect("Filter by Sentiment", df['Sentiment'].unique(), default=df['Sentiment'].unique())
    
    filtered_df = df[(df['Risk'].isin(risk_filter)) & 
                     (df['Symptom_Category'].isin(symptom_filter)) & 
                     (df['Sentiment'].isin(sentiment_filter))]
    
    # --- SUMMARY CARDS ---
    st.subheader("📊 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reports", len(filtered_df))
    col2.metric("High Risk ⚠️", len(filtered_df[filtered_df['Risk']=="High"]))
    col3.metric("Medium Risk ⚡", len(filtered_df[filtered_df['Risk']=="Medium"]))
    col4.metric("Low Risk ✅", len(filtered_df[filtered_df['Risk']=="Low"]))
    
    # --- TABS FOR VISUALS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sentiment", "Symptoms", "VBar Graphs", "Trends", "Word Cloud"])
    
    # --- SENTIMENT DOUGHNUT CHART ---
    with tab1:
        st.markdown("### Sentiment Distribution")
        fig_sent = px.pie(filtered_df, names="Sentiment", color="Sentiment",
                          color_discrete_map={"Positive":"green","Neutral":"gray","Negative":"red"},
                          hole=0.4)
        st.plotly_chart(fig_sent, use_container_width=True)
    
    # --- SYMPTOM BAR CHART ---
    with tab2:
        st.markdown("### Symptom Categories Frequency")
        symptom_counts = filtered_df['Symptom_Category'].value_counts().reset_index()
        symptom_counts.columns = ["Symptom", "Count"]
        fig_bar = px.bar(symptom_counts, x="Symptom", y="Count", color="Count",
                         color_continuous_scale="Reds", title="Symptoms Frequency")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # --- VERTICAL BAR GRAPHS ---
    with tab3:
        st.markdown("### Additional Insights with Vertical Bar Graphs")
        col1, col2 = st.columns(2)
        
        # Symptom Frequency VBar
        with col1:
            fig_vbar_symptoms = px.bar(
                symptom_counts, x="Symptom", y="Count", color="Count",
                color_continuous_scale="Reds", title="Symptoms Frequency",
                labels={"Count":"Number of Reports", "Symptom":"Symptom Category"}
            )
            st.plotly_chart(fig_vbar_symptoms, use_container_width=True)
        
        # Risk Level VBar
        with col2:
            risk_counts = filtered_df['Risk'].value_counts().reset_index()
            risk_counts.columns = ["Risk", "Count"]
            fig_vbar_risk = px.bar(
                risk_counts, x="Risk", y="Count", color="Count",
                color_continuous_scale="Oranges", title="Reports by Risk Level",
                labels={"Count":"Number of Reports", "Risk":"Risk Level"}
            )
            st.plotly_chart(fig_vbar_risk, use_container_width=True)
    
    # --- TREND LINE CHART ---
    with tab4:
        if 'Date' in filtered_df.columns:
            st.markdown("### Severity Trend Over Time")
            filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
            trend = filtered_df.groupby(['Date','Symptom_Category']).size().reset_index(name='Count')
            fig_line = px.line(trend, x='Date', y='Count', color='Symptom_Category', markers=True,
                               title="Symptom Trend Over Time")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Add a 'Date' column in your CSV to view trends over time.")
    
    # --- WORD CLOUD ---
    with tab5:
        st.markdown("### Word Cloud of Feedback / Symptoms")
        all_text = " ".join(filtered_df['Text'].tolist())
        wc = WordCloud(width=800, height=400, background_color="#e0f7fa", colormap="Reds").generate(all_text)
        plt.figure(figsize=(12,6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    # --- TOP HIGH-RISK REPORTS ---
    st.subheader("🔴 Top High Risk Reports")
    st.dataframe(filtered_df[filtered_df['Risk']=="High"][['Text','Symptoms','Severity','Sentiment','Symptom_Category']].head(10))
    
else:
    st.info("Upload a CSV file with Text, Symptoms, and Severity columns to analyze health data!")
