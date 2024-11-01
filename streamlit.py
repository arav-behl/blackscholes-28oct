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
    .reportview-container {
        background: #f0f2f6; /* Light background for better contrast */
    }
    .main {
        background: #f0f2f6; /* Neutral background for main content */
        color: #333333; /* Dark text for readability */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #cccccc;
    }
    h1, h2, h3 {
        color: #333333; /* Dark text for headings */
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6; /* Neutral sidebar background */
    }
    .Widget>label {
        color: #333333; /* Dark text for widget labels */
    }
    .stSlider>div>div>div>div {
        background-color: #4CAF50;
    }
    .stMetric {
        background-color: #e9ecef; /* Light background for metrics */
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📈 Advanced Option Analytics")
    st.write("Developed by [Arav Behl](https://www.linkedin.com/in/arav-behl-0524a6230/)")
    
    # Input parameters
    current_price = st.number_input("Current Asset Price", value=100.0, step=0.01)
    strike = st.number_input("Strike Price", value=100.0, step=0.01)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0, step=0.01)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, step=0.01)
    dividend_yield = st.number_input("Dividend Yield", value=0.0, step=0.01)
    call_purchase_price = st.number_input("Call Purchase Price", value=10.0, step=0.01)
    put_purchase_price = st.number_input("Put Purchase Price", value=5.0, step=0.01)
    
    # New input for market price
    market_price = st.number_input("Current Market Price of the Option", value=10.0, step=0.01)
    
    st.markdown("---")
    st.subheader("Heatmap Parameters")
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    vol_max = st.slider('Max Volatility', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 20)
    vol_range = np.linspace(vol_min, vol_max, 20)

# Main content
st.title("Advanced Option Analytics Platform")

# Add a concise description for users
st.markdown("""
Welcome to the Advanced Option Analytics Platform! Here, you can explore and analyze options by adjusting various parameters:
- **Current Asset Price, Strike Price, Time to Maturity, Interest Rate, and Dividend Yield**: Use these inputs to calculate the Call and Put prices, along with their Greeks.
- **Heatmap Parameters**: Adjust the Min and Max Spot Price, Volatility and market price you set to visualize the P&L heatmap, showing how these factors affect option profitability.
- **Sensitivity Analysis**: Select a parameter to see how changes impact option pricing.
- **Monte Carlo Simulation**: Run simulations to estimate future asset prices and option values.
- **Implied Volatility**: Automatically calculated based on the theoretical model and market prices.

Tweak these values to gain insights into option pricing dynamics and risk management.
""")

# Calculate option prices and Greeks
bs_model = BlackScholes(time_to_maturity, strike, current_price, interest_rate, dividend_yield)
bs_model.run()
implied_vol = bs_model.calculate_implied_volatility(market_price=market_price)  # Calculate and store implied volatility

# Display key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Call Price", f"${bs_model.call_price:.2f}")
with col2:
    st.metric("Put Price", f"${bs_model.put_price:.2f}")
with col3:
    st.metric("Call Delta", f"{bs_model.call_delta:.4f}")
with col4:
    st.metric("Put Delta", f"{bs_model.put_delta:.4f}")

# Greeks and additional metrics
st.subheader("Option Greeks and Metrics")
greeks_df = pd.DataFrame({
    "Metric": ["Gamma", "Vega", "Theta", "Rho"],
    "Call": [bs_model.call_gamma, bs_model.call_vega, bs_model.call_theta, bs_model.call_rho],
    "Put": [bs_model.put_gamma, bs_model.put_vega, bs_model.put_theta, bs_model.put_rho]
})
st.table(greeks_df.set_index("Metric").style.format("{:.4f}"))

# Display Implied Volatility separately
if implied_vol is not None and not np.isnan(implied_vol):
    st.metric("Implied Volatility", f"{implied_vol:.4f}")
else:
    st.metric("Implied Volatility", "Calculation Error")

# Interactive Payoff Diagram
st.subheader("Option Payoff Diagram")
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
    template="plotly_white"  # Use a light template for better contrast
)
st.plotly_chart(fig, use_container_width=True)

# Sensitivity Analysis
st.subheader("Sensitivity Analysis")
sensitivity_params = ['Spot Price', 'Time to Maturity', 'Interest Rate']
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
    
    bs_temp.run()
    call_prices.append(bs_temp.call_price)
    put_prices.append(bs_temp.put_price)

fig = go.Figure()
fig.add_trace(go.Scatter(x=param_range, y=call_prices, mode='lines', name='Call Price'))
fig.add_trace(go.Scatter(x=param_range, y=put_prices, mode='lines', name='Put Price'))
fig.update_layout(
    title=f'Option Price Sensitivity to {selected_param}',
    xaxis_title=f'{selected_param} (relative to current)',
    yaxis_title='Option Price',
    template="plotly_white"  # Use a light template for better contrast
)
st.plotly_chart(fig, use_container_width=True)

# Heatmaps for Call and Put P&L
st.subheader("Option P&L Heatmaps")
col1, col2 = st.columns(2)

def plot_pnl_heatmap(option_type, purchase_price):
    pnl = np.zeros((len(vol_range), len(spot_range)))
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(time_to_maturity, strike, spot, vol, interest_rate, dividend_yield)
            bs_temp.run()
            if option_type == 'call':
                current_price = bs_temp.call_price
                pnl[i, j] = current_price - purchase_price  # P&L = Current Value - Purchase Price
            else:
                current_price = bs_temp.put_price
                pnl[i, j] = current_price - purchase_price

    fig = px.imshow(
        pnl,
        x=spot_range,
        y=vol_range,
        color_continuous_scale="RdYlGn",  # Red-Yellow-Green color scale
        labels=dict(x="Spot Price", y="Volatility", color="P&L"),
        title=f"{option_type.capitalize()} Option P&L",
        aspect="auto"  # Maintains aspect ratio
    )
    
    # Update layout for better visualization
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="P&L",
            tickprefix="$",
        ),
        template="plotly_white",  # Use a light template for better contrast
        xaxis_title="Spot Price",
        yaxis_title="Volatility",
    )
    
    # Add value annotations
    for i in range(len(vol_range)):
        for j in range(len(spot_range)):
            fig.add_annotation(
                x=spot_range[j],
                y=vol_range[i],
                text=f"{pnl[i, j]:.2f}",
                showarrow=False,
                font=dict(size=8, color="black")  # Dark text for better contrast
            )
    
    return fig

with col1:
    st.plotly_chart(plot_pnl_heatmap('call', call_purchase_price), use_container_width=True)

with col2:
    st.plotly_chart(plot_pnl_heatmap('put', put_purchase_price), use_container_width=True)

# Monte Carlo Simulation
st.subheader("Monte Carlo Simulation")
num_simulations = st.slider("Number of Simulations", min_value=1000, max_value=10000, value=5000, step=1000)
num_steps = 252  # Assuming daily steps for a year

def simulate_price_path(S, T, r, sigma, steps):
    dt = T / steps
    price_path = [S]
    for _ in range(steps):
        dS = S * (r * dt + sigma * np.sqrt(dt) * np.random.normal())
        S += dS
        price_path.append(S)
    return price_path

simulated_paths = [simulate_price_path(current_price, time_to_maturity, interest_rate, 0.2, num_steps) for _ in range(num_simulations)]

fig = go.Figure()
for path in simulated_paths[:100]:  # Plot first 100 paths
    fig.add_trace(go.Scatter(y=path, mode='lines', line=dict(width=1), opacity=0.3))
fig.update_layout(
    title='Monte Carlo Simulation of Asset Price Paths',
    xaxis_title='Time Steps',
    yaxis_title='Asset Price',
    template="plotly_white"  # Use a light template for better contrast
)
st.plotly_chart(fig, use_container_width=True)

# Calculate option prices based on simulated paths
final_prices = [path[-1] for path in simulated_paths]
call_payoffs = np.maximum(np.array(final_prices) - strike, 0)
put_payoffs = np.maximum(strike - np.array(final_prices), 0)

mc_call_price = np.mean(call_payoffs) * np.exp(-interest_rate * time_to_maturity)
mc_put_price = np.mean(put_payoffs) * np.exp(-interest_rate * time_to_maturity)

st.write(f"Monte Carlo Call Price: ${mc_call_price:.2f}")
st.write(f"Monte Carlo Put Price: ${mc_put_price:.2f}")

# Add a footer
st.markdown("---")
st.markdown("Developed by [Arav Behl](https://www.linkedin.com/in/arav-behl-0524a6230/) | [GitHub](https://github.com/arav-behl)")
