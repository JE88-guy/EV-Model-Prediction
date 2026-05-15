import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="CT EV Registration Forecaster",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling with dark mode support ---
st.markdown("""
<style>
    /* Force white text on dark backgrounds */
    .stApp {
        background-color: transparent;
        color: #ffffff !important;
    }
    
    /* All text elements */
    .stMarkdown, .stText, .stMetric, label, .st-emotion-cache-1v0mbdj, .st-emotion-cache-10trblm {
        color: #ffffff !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6, .stHeading {
        color: #ffffff !important;
    }
    
    /* Sidebar text */
    .css-1d391kg, .css-12oz5g7, .stSidebar .stMarkdown, .stSidebar label, .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white !important;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Forecast card */
    .forecast-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white !important;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    .forecast-card h1, .forecast-card h3, .forecast-card p {
        color: white !important;
    }
    
    /* Insight boxes */
    .insight-box {
        background-color: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
        color: white !important;
    }
    
    .insight-box strong, .insight-box * {
        color: white !important;
    }
    
    /* Metric card */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white !important;
    }
    
    .metric-card strong, .metric-card * {
        color: white !important;
    }
    
    /* Info box */
    .info-box {
        background-color: rgba(59, 130, 246, 0.2);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white !important;
    }
    
    .info-box strong, .info-box * {
        color: white !important;
    }
    
    /* Success box */
    .success-box {
        background-color: rgba(34, 197, 94, 0.2);
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white !important;
    }
    
    .success-box strong, .success-box * {
        color: white !important;
    }
    
    /* Warning box */
    .warning-box {
        background-color: rgba(234, 179, 8, 0.2);
        border-left: 4px solid #eab308;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white !important;
    }
    
    .warning-box strong, .warning-box * {
        color: white !important;
    }
    
    /* Caption and small text */
    .stCaption, caption, .st-emotion-cache-16idsys {
        color: #cccccc !important;
    }
    
    /* Input labels and values */
    .stNumberInput label, .stSelectbox label, .stSlider label {
        color: white !important;
    }
    
    /* Metric display */
    .stMetric label, .stMetric .st-emotion-cache-1wivap2 {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: white !important;
    }
    
    /* Dataframe */
    .stDataFrame, .dataframe, .stTable {
        color: white !important;
    }
    
    /* Download button */
    .stDownloadButton button {
        background-color: #4CAF50;
        color: white !important;
    }
    
    /* Sidebar specific */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Progress and spinner */
    .stSpinner > div {
        color: white !important;
    }
    
    /* Code blocks */
    code {
        color: #ffd700 !important;
        background-color: rgba(0, 0, 0, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Load the Model and Scaler with Error Handling ---
@st.cache_resource
def load_assets():
    """Load model artifacts with error handling"""
    try:
        model = joblib.load('Model/ev_rf_model.pkl')
        scaler = joblib.load('Model/scaler.pkl')
        features = joblib.load('Model/feature_names.pkl')
        return model, scaler, features
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found. Please ensure all files exist in the 'Model/' directory.\n\nError: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        st.stop()

# Load assets
model, scaler, feature_names = load_assets()

# --- 2. Helper Functions ---
def calculate_month_index(year, month, start_year=2022, start_month=1):
    """Calculate month index based on reference point"""
    return (year - start_year) * 12 + (month - start_month)

def get_season(month):
    """Determine season based on month"""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def get_quarter(month):
    """Get quarter of the year"""
    return (month - 1) // 3 + 1

def make_prediction(input_year, input_month, input_lag_1, month_index):
    """Make prediction with input validation"""
    # Validate inputs
    if input_lag_1 < 0:
        st.error("❌ Previous month registration cannot be negative.")
        return None
    if input_lag_1 > 50000:
        st.warning("⚠️ Warning: The input value is higher than typical historical volumes. Prediction may be less accurate.")
    
    # Prepare input dictionary
    input_data = {feat: 0 for feat in feature_names}
    
    # Fill numeric features
    input_data['year'] = input_year
    input_data['month_index'] = month_index
    input_data['lag_1'] = input_lag_1
    
    # Set month dummy
    month_col = f'month_{input_month}'
    if month_col in input_data:
        input_data[month_col] = 1
    
    # Convert, scale, and predict
    input_df = pd.DataFrame([input_data])[feature_names]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    return max(0, prediction)  # Ensure non-negative prediction

def create_forecast_chart(current_value, forecast_value, month_name):
    """Create a forecast comparison chart using matplotlib with white text"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set style for dark mode
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    categories = ['Previous Month', f'Forecast: {month_name}']
    values = [current_value, forecast_value]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='none', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars with white text
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(value):,}', ha='center', va='bottom', fontweight='bold', fontsize=12,
                color='white')
    
    # Calculate percentage change
    pct_change = ((forecast_value - current_value) / current_value * 100) if current_value > 0 else 0
    
    # Add a line showing the change
    ax.axhline(y=current_value, color='gray', linestyle='--', alpha=0.5, label=f'Baseline: {int(current_value):,}')
    
    # Add annotation for percentage change
    if pct_change > 0:
        ax.annotate(f'↑ +{pct_change:.1f}%', 
                   xy=(1, forecast_value), 
                   xytext=(1.1, forecast_value + (forecast_value * 0.05)),
                   fontsize=12, fontweight='bold', color='#22c55e',
                   arrowprops=dict(arrowstyle='->', color='#22c55e', lw=1.5))
    else:
        ax.annotate(f'↓ {pct_change:.1f}%', 
                   xy=(1, forecast_value), 
                   xytext=(1.1, forecast_value + (forecast_value * 0.05)),
                   fontsize=12, fontweight='bold', color='#ef4444',
                   arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.5))
    
    # Set colors for text and lines - ALL WHITE
    ax.set_ylabel('Number of Registrations', fontsize=12, fontweight='bold', color='white')
    ax.set_title(f'EV Registration Forecast: {month_name}', fontsize=14, fontweight='bold', pad=20, color='white')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.2, color='gray')
    
    # Set all text to white
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    # Set spine colors
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.5)
    
    plt.tight_layout()
    return fig

def create_trend_forecast(input_lag_1, prediction, months=6):
    """Create a simple trend forecast for next months with white text"""
    # Simple exponential smoothing for visualization
    alpha = 0.3
    trend = [input_lag_1]
    for i in range(months):
        next_val = alpha * prediction + (1 - alpha) * trend[-1]
        trend.append(next_val)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set style for dark mode
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    x_labels = ['Current'] + [f'Month {i+1}' for i in range(months)]
    
    ax.plot(x_labels, trend, marker='o', linewidth=2.5, markersize=8, 
            color='#2E86AB', label='Projected Trend')
    
    # Highlight the forecast point
    ax.plot(x_labels[1], trend[1], 'o', markersize=12, color='#ff7f0e', 
            label=f'Forecast: {int(prediction):,}')
    
    # Add value labels with white text
    for i, (x, y) in enumerate(zip(x_labels, trend)):
        ax.annotate(f'{int(y):,}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9, color='white')
    
    # Set all text to white
    ax.set_xlabel('Time Period', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('Projected Registrations', fontsize=12, fontweight='bold', color='white')
    ax.set_title('6-Month Trend Projection', fontsize=14, fontweight='bold', pad=20, color='white')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2, color='gray')
    ax.tick_params(colors='white')
    
    # Set spine colors
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.5)
    
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    return fig

def create_seasonal_comparison(input_month, prediction):
    """Create seasonal comparison chart with white text"""
    # Typical seasonal factors (example values - adjust based on your domain knowledge)
    seasonal_factors = {
        1: 0.85, 2: 0.88, 3: 0.92, 4: 0.95, 5: 1.0, 6: 1.05,
        7: 1.08, 8: 1.1, 9: 1.05, 10: 0.98, 11: 0.92, 12: 0.88
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set style for dark mode
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    months = list(range(1, 13))
    month_names = [datetime(2000, m, 1).strftime('%b') for m in months]
    base_values = [prediction / seasonal_factors[input_month] * seasonal_factors[m] for m in months]
    
    colors = ['#ff7f0e' if m == input_month else '#1f77b4' for m in months]
    bars = ax.bar(month_names, base_values, color=colors, edgecolor='none', linewidth=1, alpha=0.8)
    
    # Add value labels with white text
    for bar, value in zip(bars, base_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(value):,}', ha='center', va='bottom', fontsize=8, rotation=45, color='white')
    
    # Set all text to white
    ax.set_xlabel('Month', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('Estimated Registrations', fontsize=12, fontweight='bold', color='white')
    ax.set_title('Seasonal Comparison - Estimated Monthly Pattern', fontsize=14, fontweight='bold', pad=20, color='white')
    ax.grid(axis='y', alpha=0.2, color='gray')
    ax.tick_params(colors='white')
    
    # Set spine colors
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.5)
    
    # Highlight the selected month
    ax.axvspan(input_month - 1.5, input_month - 0.5, alpha=0.2, color='orange')
    
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    return fig

# --- 3. Sidebar Configuration ---
st.sidebar.title("⚙️ Forecast Settings")

# Create tabs in sidebar for better organization
settings_tab, info_tab = st.sidebar.tabs(["📊 Input Parameters", "ℹ️ About"])

with settings_tab:
    # Date selection with better UI
    col1, col2 = st.columns(2)
    with col1:
        input_year = st.number_input(
            "📅 Year",
            min_value=2024,
            max_value=2030,
            value=datetime.now().year + 1,
            help="Select the target year for forecast"
        )
    with col2:
        input_month = st.selectbox(
            "📆 Month",
            options=list(range(1, 13)),
            format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
            index=5,
            help="Select the target month"
        )
    
    # Dynamic month name
    month_name = datetime(2000, input_month, 1).strftime('%B')
    
    # Historical volume input with better UI
    st.markdown("---")
    st.subheader("📈 Historical Data")
    
    # Add quick suggestions based on month/season
    season = get_season(input_month)
    avg_ranges = {
        "Winter": "300-600",
        "Spring": "500-800", 
        "Summer": "600-900",
        "Fall": "450-750"
    }
    
    st.caption(f"💡 Typical {season} range: {avg_ranges[season]} units")
    
    input_lag_1 = st.number_input(
        "🚗 Previous Month Registration Volume",
        min_value=0,
        value=500,
        step=50,
        format="%d",
        help="Enter the actual registration volume from the previous month"
    )

with info_tab:
    st.markdown("""
    ### 🤖 About This Tool
    
    **Model:** Random Forest Regressor
    
    **Features used:**
    - 📅 Year & Month (seasonality)
    - 📈 Previous month registrations (momentum)
    - 🎯 Month indicators
    
    ### 📊 Interpretation
    
    - **Growth:** Forecast > Previous month
    - **Decline:** Forecast < Previous month
    - **Stable:** Within ±5% change
    
    ### ⚠️ Limitations
    
    - Predictions assume similar market conditions
    - External factors (incentives, policies) not included
    - Accuracy decreases for distant forecasts
    """)

# Calculate month index
month_index = calculate_month_index(input_year, input_month)

# --- 4. Main UI ---
st.title("🔌 Connecticut EV Registration Forecaster")

# Use styled markdown with proper background handling
st.markdown("""
<div class="info-box">
    <strong>🎯 Smart Forecasting</strong> — Powered by machine learning to predict electric vehicle adoption trends in Connecticut.
</div>
""", unsafe_allow_html=True)

# Display current parameters
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📅 Target Period", f"{month_name} {input_year}")
with col2:
    st.metric("🎯 Season", season)
with col3:
    st.metric("📊 Quarter", f"Q{get_quarter(input_month)}")

st.markdown("---")

# Generate forecast button
if st.button("🚀 Generate Forecast", use_container_width=True):
    with st.spinner("🤖 Analyzing patterns and generating forecast..."):
        result = make_prediction(input_year, input_month, input_lag_1, month_index)
        
        if result is not None:
            # Create main forecast display
            st.markdown("---")
            
            # Results in columns - now only 2 columns since we removed confidence
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="forecast-card">
                    <h3 style="margin: 0 0 0.5rem 0; color: white;">📊 Predicted Registrations</h3>
                    <h1 style="font-size: 3rem; margin: 0; color: white;">{int(result):,}</h1>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; color: white;">for {month_name} {input_year}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                change = result - input_lag_1
                change_pct = (change / input_lag_1 * 100) if input_lag_1 > 0 else 0
                if change > 0:
                    st.metric("📈 Change from Previous", f"+{int(change):,}", f"+{change_pct:.1f}%", delta_color="normal")
                else:
                    st.metric("📉 Change from Previous", f"{int(change):,}", f"{change_pct:.1f}%", delta_color="inverse")
            
            st.markdown("---")
            
            # Visualization tabs
            tab1, tab2, tab3 = st.tabs(["📊 Comparison Chart", "📈 Trend Projection", "📅 Seasonal Pattern"])
            
            with tab1:
                st.pyplot(create_forecast_chart(input_lag_1, result, f"{month_name} {input_year}"))
            
            with tab2:
                st.pyplot(create_trend_forecast(input_lag_1, result))
            
            with tab3:
                st.pyplot(create_seasonal_comparison(input_month, result))
                st.caption("💡 This chart shows estimated seasonal patterns. Your selected month is highlighted in orange.")
            
            # Dynamic insights
            st.markdown("---")
            st.subheader("💡 Key Insights & Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if change > 0:
                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ Market Analysis</strong><br>
                        • Growth trend detected<br>
                        • Positive momentum in EV adoption<br>
                        • Seasonal factors favorable
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ Market Analysis</strong><br>
                        • Decline trend detected<br>
                        • Review market conditions<br>
                        • Monitor external factors
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if change > 0:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>✅ Recommendations</strong><br>
                        • Increase inventory by {int(change * 0.2):,} units<br>
                        • Ramp up marketing in {month_name}<br>
                        • Prepare for {change_pct:.1f}% growth
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        <strong>⚠️ Recommendations</strong><br>
                        • Reduce inventory by {int(abs(change) * 0.2):,} units<br>
                        • Focus on retention marketing<br>
                        • Adjust expectations by {abs(change_pct):.1f}%
                    </div>
                    """, unsafe_allow_html=True)
            
            # Download option
            st.markdown("---")
            forecast_data = pd.DataFrame({
                'Metric': ['Target Year', 'Target Month', 'Season', 'Quarter', 'Previous Volume', 
                          'Forecasted Volume', 'Absolute Change', 'Percentage Change'],
                'Value': [input_year, month_name, season, f"Q{get_quarter(input_month)}", 
                         f"{input_lag_1:,}", f"{int(result):,}", f"{int(change):,}", 
                         f"{change_pct:.1f}%"]
            })
            
            csv = forecast_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Report (CSV)",
                data=csv,
                file_name=f"ev_forecast_{input_year}_{input_month:02d}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    # Welcome message and instructions
    st.markdown("""
    <div class="info-box">
        <strong>👈 Get Started</strong> — Adjust the parameters in the sidebar and click 'Generate Forecast' to see predictions.
    </div>
    """, unsafe_allow_html=True)
    
    # Show how it works
    st.markdown("### 📊 How It Works")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <strong>1️⃣ Input Parameters</strong><br><br>
            • Select year & month<br>
            • Enter previous month volume<br>
            • System adjusts for seasonality
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <strong>2️⃣ AI Processing</strong><br><br>
            • Random Forest analysis<br>
            • Pattern recognition<br>
            • Seasonality adjustment
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <strong>3️⃣ Get Insights</strong><br><br>
            • Immediate forecast<br>
            • Trend visualization<br>
            • Actionable recommendations
        </div>
        """, unsafe_allow_html=True)
    
    # Add a sample chart to show what to expect
    st.markdown("---")
    st.markdown("### 📈 Example Forecast Visualization")
    
    # Create a sample chart with white text
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    sample_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    sample_values = [450, 520, 580, 610, 650, 720]
    ax.plot(sample_months, sample_values, marker='o', linewidth=2, color='#2E86AB')
    ax.fill_between(sample_months, sample_values, alpha=0.2, color='#2E86AB')
    
    # Set all text to white
    ax.set_ylabel('Registrations', color='white')
    ax.set_title('Sample EV Registration Trend', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='gray')
    
    # Set spine colors
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.5)
    
    st.pyplot(fig)
    st.caption("Above is just an example - your actual forecast will appear when you generate it!")

# --- Footer ---
st.markdown("---")
st.caption(f"🔋 Connecticut EV Registration Forecaster | Powered by Random Forest | Last updated: {datetime.now().strftime('%B %d, %Y')}")
