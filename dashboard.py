import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analyst_core import load_data, clean_dataframe

# Set page config
st.set_page_config(page_title="Instagram Analytics Dashboard", layout="wide")

st.title("📸 Instagram Analytics Dashboard")
st.markdown("Interactive analysis of your Instagram data.")

# Load Data
@st.cache_data
def get_data():
    df = load_data("Instagram_Final_Data.xlsx")
    df = clean_dataframe(df)
    return df

try:
    df = get_data()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar
st.sidebar.header("Filter Data")
min_views = st.sidebar.slider("Minimum Views", 0, int(df['views'].max()), 0)
filtered_df = df[df['views'] >= min_views]

# Key Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Posts", len(filtered_df))
col2.metric("Total Views", f"{filtered_df['views'].sum():,.0f}")
col3.metric("Avg Likes", f"{filtered_df['likes'].mean():,.0f}")

# Charts
st.markdown("## 📊 Visualizations")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Share of Views (Top 5 Creators)")
    user_views = filtered_df.groupby('Username')['views'].sum().sort_values(ascending=False)
    top_5 = user_views.head(5)
    others = pd.Series([user_views.iloc[5:].sum()], index=['Others'])
    pie_data = pd.concat([top_5, others])
    
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    st.pyplot(fig1)

with col_chart2:
    st.subheader("Top 10 Creators by Followers")
    user_followers = filtered_df.groupby('Username')['Username_Followers'].max().sort_values(ascending=False).head(10)
    
    fig2, ax2 = plt.subplots()
    sns.barplot(x=user_followers.values, y=user_followers.index, hue=user_followers.index, palette='viridis', legend=False, ax=ax2)
    ax2.set_xlabel("Followers")
    st.pyplot(fig2)

# Scatter Plot
st.subheader("Engagement: Views vs Likes")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=filtered_df, x='views', y='likes', alpha=0.6, ax=ax3)
ax3.set_title("Views vs Likes")
st.pyplot(fig3)

# Data Table
st.markdown("## 📋 Raw Data")
st.dataframe(filtered_df)

# --- Viral Predictor Section ---
st.markdown("---")
st.markdown("## 🤖 Viral Predictor (Machine Learning)")

from analyst_core import train_viral_predictor
import numpy as np

model, score, X_test, y_test, y_pred = train_viral_predictor(df)

if model:
    col_pred1, col_pred2 = st.columns([1, 2])
    
    with col_pred1:
        st.subheader("Predict Views")
        followers_input = st.number_input("Enter Follower Count:", min_value=0, value=10000, step=1000)
        
        if st.button("Predict Viral Potential"):
            prediction = model.predict(np.array([[followers_input]]))[0]
            st.metric("Predicted Views", f"{prediction:,.0f}")
            
            # Context
            st.info(f"Model Accuracy (R²): {score:.2f}")
            if score < 0.1:
                st.warning("Warning: The correlation between followers and views is weak in this dataset. Predictions may be inaccurate.")
                
    with col_pred2:
        st.subheader("Actual vs Predicted Performance")
        fig4, ax4 = plt.subplots()
        ax4.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
        ax4.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Trend')
        ax4.set_xlabel("Followers")
        ax4.set_ylabel("Views")
        ax4.legend()
        st.pyplot(fig4)
else:
    st.warning("Not enough data to train the predictor model (need at least 10 rows).")
