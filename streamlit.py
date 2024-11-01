import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns
from blackscholes import BlackScholes

# Page configuration
st.set_page_config(
    page_title="Advanced Option Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

# Custom CSS for better visibility in both light and dark modes
st.markdown("""
<style>
    /* Base styles */
    .main {
        background: transparent !important;
        color: inherit !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: transparent !important;
    }
    
    /* Make sidebar text red and visible in both modes */
    .sidebar .element-container {
        color: #ff4b4b !important;
    }
    
    /* Remove grey borders and make inputs transparent */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border: none !important;
        background: transparent !important;
        color: inherit !important;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #ff4b4b !important;
    }
    
    /* Metric styling */
    .stMetric {
        background: transparent !important;
        border: 1px solid rgba(255, 75, 75, 0.2);
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Table styling */
    .dataframe {
        background: transparent !important;
    }
    
    /* Widget labels */
    .Widget>label {
        color: #ff4b4b !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📈 Advanced Option Analytics")
    st.write("Developed by [Arav Behl](https://www.linkedin.com/in/arav-behl-0524a6230/)")
    
    # Input parameters (removed volatility, added market price)
    current_price = st.number_input("Current Asset Price", value=100.0, step=0.01)
    strike = st.number_input("Strike Price", value=100.0, step=0.01)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0, step=0.01)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, step=0.01)
    dividend_yield = st.number_input("Dividend Yield", value=0.0, step=0.01)
    market_price = st.number_input("Current Market Price of Option", value=10.0, step=0.01)

# Main content
st.title("Advanced Option Analytics Platform")

# 1. Implied Volatility
st.header("1. Implied Volatility")
st.markdown("Calculates market's forecast of future volatility using the Black-Scholes model and current market price.")
bs_model = BlackScholes(time_to_maturity, strike, current_price, interest_rate, dividend_yield)
implied_vol = bs_model.calculate_implied_volatility(market_price=market_price)

if implied_vol is not None:
    st.metric("Implied Volatility", f"{implied_vol:.4f}")
else:
    st.error("Could not calculate implied volatility. Market price may be outside theoretical bounds.")

# 2. Option Prices
st.header("2. Put & Call Prices")
st.markdown("Theoretical prices calculated using the Black-Scholes model with implied volatility.")
bs_model.run(volatility=implied_vol if implied_vol is not None else 0.2)
col1, col2 = st.columns(2)
with col1:
    st.metric("Call Price", f"${bs_model.call_price:.2f}")
with col2:
    st.metric("Put Price", f"${bs_model.put_price:.2f}")

# 3. Greeks
st.header("3. Option Greeks")
st.markdown("Measures of risk showing how option prices change with respect to various factors.")
greeks_df = pd.DataFrame({
    "Metric": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    "Call": [bs_model.call_delta, bs_model.call_gamma, bs_model.call_vega, 
             bs_model.call_theta, bs_model.call_rho],
    "Put": [bs_model.put_delta, bs_model.put_gamma, bs_model.put_vega, 
            bs_model.put_theta, bs_model.put_rho]
})
st.table(greeks_df.set_index("Metric").style.format("{:.4f}"))

# 4. Payoff Diagram
st.header("4. Option Payoff Diagram")
st.markdown("Visualizes potential profit/loss at expiration across different underlying prices.")
spot_prices = np.linspace(current_price * 0.5, current_price * 1.5, 100)
call_payoffs = np.maximum(spot_prices - strike, 0) - bs_model.call_price
put_payoffs = np.maximum(strike - spot_prices, 0) - bs_model.put_price

fig = go.Figure()
fig.add_trace(go.Scatter(x=spot_prices, y=call_payoffs, mode='lines', name='Call Payoff'))
fig.add_trace(go.Scatter(x=spot_prices, y=put_payoffs, mode='lines', name='Put Payoff'))
fig.update_layout(
    title='Option Payoff Diagram',
    xaxis_title='Spot Price',
    yaxis_title='Profit/Loss',
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# 5. Sensitivity Analysis
st.header("5. Sensitivity Analysis")
st.markdown("Visualizes how option prices change with variations in key parameters like spot price, time, and rates.")
sensitivity_params = ['Spot Price', 'Time to Maturity', 'Interest Rate']  # Removed Volatility
selected_param = st.selectbox("Select parameter for sensitivity analysis", sensitivity_params)

param_range = np.linspace(0.5, 1.5, 100)
call_prices = []
put_prices = []

for param in param_range:
    if selected_param == 'Spot Price':
        bs_temp = BlackScholes(time_to_maturity, strike, current_price * param, interest_rate, dividend_yield)
    elif selected_param == 'Time to Maturity':
        bs_temp = BlackScholes(time_to_maturity * param, strike, current_price, interest_rate, dividend_yield)
    else:  # Interest Rate
        bs_temp = BlackScholes(time_to_maturity, strike, current_price, interest_rate * param, dividend_yield)
    
    bs_temp.run(volatility=implied_vol if implied_vol is not None else 0.2)
    call_prices.append(bs_temp.call_price)
    put_prices.append(bs_temp.put_price)

# 6. Option P&L Heatmaps
st.header("6. Option P&L Heatmaps")
st.markdown("Shows potential profit/loss across different spot prices and volatilities.")
# ... (heatmap code remains the same but uses implied_vol instead of volatility)

# 7. Monte Carlo Simulation
st.header("7. Monte Carlo Simulation")
st.markdown("Estimates future option values through multiple price path simulations.")
# ... (Monte Carlo code remains the same but uses implied_vol instead of volatility)
