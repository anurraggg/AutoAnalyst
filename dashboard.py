import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from analyst_core import load_data, clean_dataframe, train_viral_predictor

# Set page config
st.set_page_config(page_title="Instagram Analytics Dashboard", layout="wide", page_icon="📸")

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background - Target the root container */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Alternative Main Background Selector */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        background-attachment: fixed !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 12, 41, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Metric Cards (Glassmorphism) */
    div[data-testid="stMetric"], .stMetric {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 20px !important;
        border-radius: 15px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    }
    
    div[data-testid="stMetric"]:hover, .stMetric:hover {
        border-color: #E1306C !important;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Metric Value Color */
    [data-testid="stMetricValue"] {
        color: #E1306C !important;
        font-size: 2rem !important;
    }
    
    /* Metric Label Color */
    [data-testid="stMetricLabel"] {
        color: #e0e0e0 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent !important;
        border-radius: 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #aaa !important;
        font-weight: 600;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #E1306C !important;
        border-bottom: 3px solid #E1306C !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #E1306C, #C13584) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 10px 25px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(225, 48, 108, 0.3) !important;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(225, 48, 108, 0.5) !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FAFAFA !important;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Plotly Chart Container */
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

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
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar
st.sidebar.header("🔍 Filters")
min_views = st.sidebar.number_input("Minimum Views", min_value=0, max_value=int(df['views'].max()), value=0, step=1000)
filtered_df = df[df['views'] >= min_views]

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("This dashboard uses **Machine Learning** to predict viral potential and **Plotly** for interactive visualizations.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔎 Deep Dive", "🤖 Viral Predictor"])

with tab1:
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", len(filtered_df))
    col2.metric("Total Views", f"{filtered_df['views'].sum():,.0f}")
    col3.metric("Total Likes", f"{filtered_df['likes'].sum():,.0f}")
    col4.metric("Avg Engagement", f"{filtered_df['likes'].mean():,.0f}")

    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Share of Views (Top 5 Creators)")
        user_views = filtered_df.groupby('Username')['views'].sum().sort_values(ascending=False)
        top_5 = user_views.head(5)
        others = pd.Series([user_views.iloc[5:].sum()], index=['Others'])
        pie_data = pd.concat([top_5, others]).reset_index()
        pie_data.columns = ['Username', 'Views']
        
        fig1 = px.pie(pie_data, values='Views', names='Username', hole=0.4, 
                      color_discrete_sequence=px.colors.sequential.RdBu)
        fig1.update_layout(showlegend=True)
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.subheader("Top 10 Creators by Followers")
        user_followers = filtered_df.groupby('Username')['Username_Followers'].max().sort_values(ascending=False).head(10).reset_index()
        
        fig2 = px.bar(user_followers, x='Username_Followers', y='Username', orientation='h',
                      color='Username_Followers', color_continuous_scale='Viridis')
        fig2.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Engagement Analysis: Views vs Likes")
    
    # Scatter Plot
    # Ensure 'comments' are non-negative and fill NaNs for size
    filtered_df['comments_size'] = filtered_df['comments'].fillna(0).clip(lower=0)
    
    fig3 = px.scatter(filtered_df, x='views', y='likes', size='comments_size', hover_data=['Username', 'URL'],
                      color='views', color_continuous_scale='Plasma', log_x=True, log_y=True)
    fig3.update_layout(title="Views vs Likes (Size = Comments)", xaxis_title="Views (Log Scale)", yaxis_title="Likes (Log Scale)")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("### 📋 Raw Data")
    st.dataframe(filtered_df, use_container_width=True)

with tab3:
    st.markdown("## 🤖 Viral Predictor (Machine Learning)")
    st.markdown("Predict the potential views of a post based on follower count using a **Linear Regression** model.")

    model, score, X_test, y_test, y_pred = train_viral_predictor(df)

    if model:
        col_pred1, col_pred2 = st.columns([1, 2])
        
        with col_pred1:
            st.markdown("### 🔮 Predict")
            followers_input = st.number_input("Enter Follower Count:", min_value=0, value=10000, step=1000)
            
            if st.button("Predict Viral Potential", type="primary"):
                prediction = model.predict(np.array([[followers_input]]))[0]
                st.metric("Predicted Views", f"{prediction:,.0f}")
                
                # Context
                st.info(f"Model Accuracy (R²): {score:.2f}")
                if score < 0.1:
                    st.warning("Warning: Weak correlation detected. Predictions may be inaccurate.")
                    
        with col_pred2:
            st.markdown("### 📈 Model Performance")
            
            # Create a dataframe for plotting
            plot_data = pd.DataFrame({
                'Followers': X_test['Username_Followers'],
                'Actual Views': y_test,
                'Predicted Trend': y_pred
            })
            
            fig4 = px.scatter(plot_data, x='Followers', y='Actual Views', opacity=0.6, title="Actual vs Predicted")
            # Add trend line
            fig4.add_traces(px.line(plot_data, x='Followers', y='Predicted Trend').data[0])
            fig4.data[1].line.color = 'red'
            
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Not enough data to train the predictor model (need at least 10 rows).")
