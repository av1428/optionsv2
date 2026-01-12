import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
import os
from datetime import datetime


# Fix for Streamlit Cloud: Ensure root directory is in sys.path
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

try:
    from backend.app.strategies.scanner import options_scanner
    from backend.app.core.config import settings
except ImportError as e:
    import streamlit as st
    st.error(f"Import Error: {e}")
    st.info("Check requirements.txt and file structure.")
    st.stop()
except Exception as e:
    import streamlit as st
    st.error(f"Critical System Error: {e}")
    import traceback
    st.text(traceback.format_exc())
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="Arun's | S3-V2",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize Session State ---
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# --- Custom CSS (Institutional Dark Mode) ---
st.markdown("""
<style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main App Background */
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
    }
    
    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    
    /* --- Buttons (Blue Gradient) --- */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.5) !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.6) !important;
    }
    
    /* --- DROPDOWNS: The Scorched Earth Fix --- */
    /* Target every possible container for the popup menu */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"],
    ul[data-baseweb="menu"] {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
    }
    
    /* Options inside the list */
    li[role="option"],
    div[role="option"],
    li[data-baseweb="option"] {
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }
    
    /* Option Text */
    div[data-baseweb="menu"] div,
    ul[role="listbox"] div {
         color: #f8fafc !important;
    }
    
    /* Hover/Focus States */
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[role="option"]:hover,
    div[role="option"][aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }

    /* SelectBox Main Container */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-color: #334155 !important;
    }
    
    /* --- NUMBER INPUTS --- */
    /* Main Input Box */
    div[data-baseweb="input"] > div {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-color: #334155 !important;
    }
    .stNumberInput input {
        color: #f8fafc !important;
    }
    /* Step Buttons (+/-) */
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        background-color: #1e293b !important;
        border-color: #334155 !important;
    }
    /* Icons inside Step Buttons */
    button[data-testid="stNumberInputStepDown"] svg,
    button[data-testid="stNumberInputStepUp"] svg {
        fill: #f8fafc !important;
        color: #f8fafc !important;
    }
    
    /* --- EXPANDERS & CODE BLOCKS --- */
    div[data-testid="stExpander"] {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }
    .streamlit-expanderContent {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: none !important;
    }
    /* Code Blocks (Pre/Code) */
    pre, code, .stCode {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border-radius: 4px;
    }
    
    /* --- DATAFRAMES --- */
    div[data-testid="stDataFrame"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
    }
    div[data-testid="stDataFrame"] th {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        text-align: center !important;
    }
    div[data-testid="stDataFrame"] td {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    /* --- METRICS --- */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
    
    /* --- Badges --- */
    .badge-safe {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        color: #10b981;
        padding: 4px 12px;
        border-radius: 999px; font-weight: 600; font-size: 0.85rem;
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        color: #ef4444;
        padding: 4px 12px;
        border-radius: 999px; font-weight: 600; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def plot_payoff(strategy_name, spot_price):
    """
    Generate a generic payoff diagram for the selected strategy.
    Actual calculation would use real premiums/strikes.
    """
    fig = go.Figure()
    price_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 100)
    profit = np.zeros_like(price_range)
    
    if "Iron Condor" in strategy_name:
        # Mock Iron Condor Payoff
        short_put = spot_price * 0.95
        short_call = spot_price * 1.05
        premium = 2.0
        width = 5.0
        
        for i, p in enumerate(price_range):
            if short_put < p < short_call:
                profit[i] = premium
            elif p < short_put:
                profit[i] = premium - (short_put - p)
            else:
                profit[i] = premium - (p - short_call)
            # Max loss cap
            if profit[i] < (premium - width): profit[i] = premium - width

    elif "Credit Spread" in strategy_name:
        # Mock Bull Put Spread
        short_strike = spot_price * 0.98
        premium = 1.0
        width = 2.0
        for i, p in enumerate(price_range):
            if p > short_strike:
                profit[i] = premium
            else:
                profit[i] = premium - (short_strike - p)
            if profit[i] < (premium - width): profit[i] = premium - width

    fig.add_trace(go.Scatter(x=price_range, y=profit, mode='lines', 
                             line=dict(color='#00ff88' if profit.max() > 0 else '#ff4b4b', width=3),
                             fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'))
    
    fig.add_vline(x=spot_price, line_dash="dash", line_color="white", annotation_text="Spot")
    fig.add_hline(y=0, line_color="#555")
    
    fig.update_layout(
        title=f"Theoretic Payoff Model: {strategy_name} (Educational View)",
        xaxis_title="Stock Price at Exp",
        yaxis_title="P/L ($)",
        template="plotly_dark",
        plot_bgcolor="#1a1c24",
        paper_bgcolor="#1a1c24",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚛️ Options AI Config")
    risk_mode = st.selectbox("Strategy Mode", 
        ["Income (Iron Condor)", "Directional (Credit Spreads)", "Aggressive (Naked Puts)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### Portfolio")
    account_value = st.number_input("Capital (CAD)", value=6000, step=500)
    target_return = st.slider("Target Return (Annual)", 20, 200, 100, format="%d%%")
    
    st.markdown("---")
    with st.expander("Watchlist Settings"):
        watchlist_html = f"""
        <div style='background-color: #0f172a; padding: 10px; border-radius: 4px; border: 1px solid #334155; font-family: monospace; color: #f8fafc;'>
            {'<br>'.join(settings.WATCHLIST)}
        </div>
        """
        st.markdown(watchlist_html, unsafe_allow_html=True)

# --- Main Layout ---
col_header, col_logo = st.columns([4, 1])
with col_header:
    st.title("Market Scanner")
    st.caption(f"Strategy: {risk_mode} • Data: Yahoo Finance API")

# --- Tabs ---
tab_scanner, tab_backtest = st.tabs(["📡 Live Scanner", "⏳ Strategy Backtest"])

with tab_scanner:
    # Top Metrics Row
    # Top Metrics Row (Live Data)
    from backend.app.services.yfinance_service import yfinance_service
    
    spy_quote = yfinance_service.get_quote("SPY")
    vix_quote = yfinance_service.get_quote("^VIX")
    
    spy_price = f"${spy_quote['price']:.2f}" if spy_quote else "N/A"
    spy_change = f"{spy_quote['change_percent']:.2f}%" if spy_quote else "0.00%"
    
    vix_price = f"{vix_quote['price']:.2f}" if vix_quote else "N/A"
    vix_change = f"{vix_quote['change_percent']:.2f}%" if vix_quote else "0.00%"
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("SPY Price", spy_price, spy_change)
    with m2:
        st.metric("VIX Level", vix_price, vix_change, delta_color="inverse")
    with m3:
        # Dynamic Signal Count (based on last scan or default)
        signal_count = len(st.session_state.scan_results) if st.session_state.scan_results else 0
        st.metric("Active Signals", str(signal_count), "Scan to Update")
    with m4:
        st.metric("Est. Buying Power", f"${account_value * 2:,.0f}", "2:1 Margin")

    st.markdown("---")
    
    # --- Scanner Section ---
# --- Scanner Section ---

# Initialize Session State for Results
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

if st.button("🚀 Run Live Scan", use_container_width=True):
    with st.spinner("Analyzing Market Structure & Volatility..."):
        try:
            results = options_scanner.scan_opportunities()
            st.session_state.scan_results = results  # Persist results
        except Exception as e:
            st.error(f"Scan interrupted: {e}")
            st.code(str(e))

# Display Logic (Depend on Session State)
if st.session_state.scan_results:
    results = st.session_state.scan_results
    
    if not results:
        st.info("No high-probability setups found right now. Market might be choppy.")
    else:
        # Layout: Table on Left, Details on Right
        df = pd.DataFrame(results)
        
        # --- Visual Enhancement: Add Icons & Styling Data ---
        # Add Trend Icons
        df['trend'] = df['trend'].apply(lambda x: f"🟢 {x}" if x == 'Uptrend' else (f"🔴 {x}" if x == 'Downtrend' else f"⚪ {x}"))
        
        # Add Strategy Icons & DTE
        strategy_icon_map = {
            "Iron Condor": ("🦅 Iron Condor", "30-45 Days"),
            "Put Credit Spread": ("🐂 Put Credit Spread", "21-30 Days"),
            "Call Credit Spread": ("🐻 Call Credit Spread", "21-30 Days")
        }
        
        # Apply Mapping
        df['strategy_display'] = df['recommended_strategy'].map(lambda x: strategy_icon_map.get(x, (x, "30 Days"))[0])
        df['dte_recommendation'] = df['recommended_strategy'].map(lambda x: strategy_icon_map.get(x, (x, "30 Days"))[1])
        
        
        # Custom Sentiment Formatting (Green/Blue/Amber/Error)
        def format_sentiment_label(row):
            label = row.get('sentiment_label', 'Neutral')
            score = row.get('sentiment_score', 0)
            
            if "API Error" in label:
                return "⚠️ API Error"
            elif "No News" in label:
                return "⚪ No News"
            elif "Bullish" in label:
                return f"🟢 {label} ({score:.2f})"
            elif "Bearish" in label:
                return f"🔴 {label} ({score:.2f})"
            else:
                return f"🔵 {label} ({score:.2f})"
        
        df['sentiment_display'] = df.apply(format_sentiment_label, axis=1)
        
        # Apply Pandas Styling for Sentiment Bar (Green/Red)
        # Note: We use st.dataframe(styled_df)
        styled_df = df.style.bar(
            subset=['sentiment_score'],
            align='mid',
            color=['#ef4444', '#10b981'],
            vmin=-0.35,
            vmax=0.35
        )

        # Layout: Full Width Table, followed by Details
        st.subheader("📋 Signal Board")
        st.dataframe(
            styled_df,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", help="Stock Symbol", validate="^[A-Z0-9]+$"),
                "spot_price": st.column_config.NumberColumn(
                    "Price",
                    help="Last Traded Price",
                    min_value=0,
                    max_value=10000,
                    step=0.01,
                    format="$%.2f",
                ),
                "strategy_bias": st.column_config.TextColumn(
                    "Bias",
                    help="Directional Assumption",
                ),
                "strategy_display": st.column_config.TextColumn(
                    "Strategy",
                    help="Suggested Trading Type",
                    width="medium",
                ),
                "dte_recommendation": st.column_config.TextColumn(
                    "Rec. DTE",
                    help="Recommended Days to Expiration",
                ),
                "trend": st.column_config.TextColumn(
                    "Trend",
                    help="50-Day SMA Trend",
                ),
                "earnings_alert": st.column_config.TextColumn(
                    "Earnings",
                    help="Warning if < 45 Days to Earnings",
                ),
                "support": st.column_config.NumberColumn(
                    "Support",
                    help="Support Level",
                    format="$%.2f",
                ),
                "resistance": st.column_config.NumberColumn(
                    "Resistance",
                    help="Resistance Level",
                    format="$%.2f",
                ),
                "sentiment_score": st.column_config.NumberColumn(
                    "Sentiment Score",
                    help="News Sentiment Score (Red=Bearish, Green=Bullish)",
                    format="%.2f",
                ),
                "sentiment_display": st.column_config.TextColumn(
                    "Sentiment Label",
                    help="Sentiment Status: Bullish/Bearish or API Error",
                ),
            },
            column_order=["ticker", "spot_price", "trend", "strategy_display", "dte_recommendation", "support", "resistance", "sentiment_score", "sentiment_display", "earnings_alert"],
            use_container_width=True,
            height=400,
            hide_index=True,
        )
        
        st.markdown("---")
        st.subheader("🔍 Trade Analysis (Deep Dive)")
        
        # Uses session state implicitly to keep selection stable if keys match
        selected_ticker = st.selectbox("Select Ticker to Analyze", df['ticker'].unique())
        
        if selected_ticker:
            row = df[df['ticker'] == selected_ticker].iloc[0]
            
            # create 3 columns for the Deep Dive (Setup, Techs, Payoff)
            d_col1, d_col2, d_col3 = st.columns([1, 1, 2])
            
            with d_col1:
                st.markdown(f"#### {row['ticker']} Analysis")
                st.info(f"**Strategy**: {row['strategy_display']}")
                st.write(f"**Bias**: {row['strategy_bias']}")
                st.write(f"**Rec. DTE**: {row['dte_recommendation']}")
                
                # Earnings Badge Logic
                if "EARNINGS" in row['earnings_alert']:
                     st.markdown(f"""<div class="badge-danger">⚠️ EARNINGS ({row['earnings_date']})</div>""", unsafe_allow_html=True)
                else:
                     st.markdown(f"""<div class="badge-safe">✅ Safe ({row['earnings_date']})</div>""", unsafe_allow_html=True)
            
            with d_col2:
                st.markdown("#### 🛠️ Trade Setup")
                st.write(f"**Spot Price**: ${row['spot_price']:.2f}")
                st.write(f"**Support**: ${row['support']:.2f}")
                st.write(f"**Resistance**: ${row['resistance']:.2f}")
                
                st.markdown("##### Suggested Strikes (Nearest to S/R)")
                if "Iron Condor" in row['recommended_strategy']:
                    st.success(f"**Sell Put**: ${row['support']:.2f}")
                    st.error(f"**Sell Call**: ${row['resistance']:.2f}")
                elif "Put Credit Spread" in row['recommended_strategy']:
                    st.success(f"**Sell Put**: ${row['support']:.2f}")
                    st.caption(f"**Buy Put**: ${(row['support'] * 0.95):.2f} (Hedge)")
                elif "Call Credit Spread" in row['recommended_strategy']:
                    st.error(f"**Sell Call**: ${row['resistance']:.2f}")
                    st.caption(f"**Buy Call**: ${(row['resistance'] * 1.05):.2f} (Hedge)")
                else:
                    st.write("Custom strategy required.")
            with d_col3:
                 # Payoff Plot
                 st.plotly_chart(
                     plot_payoff(row['recommended_strategy'], row['spot_price']), 
                     use_container_width=True,
                     config={'displayModeBar': False} # Fix: Hide controls to prevent overlap
                 )

            # --- Institutional Analysis (Confluence Engine) ---
            if 'advanced_data' in row and row['advanced_data']:
                adv = row['advanced_data']
                st.markdown("---")
                st.write("") # Add spacing
                st.subheader("🏦 Institutional Analysis (Confluence Engine)")
                
                # Notes / Warnings
                if adv.get('Notes'):
                    for note in adv['Notes']:
                        if "PANIC" in note:
                            st.error(f"**RISK ALERT**: {note}")
                        else:
                            st.info(f"**INSIGHT**: {note}")

                # Dual-Tier Strategy Cards
                strat_col1, strat_col2 = st.columns(2)
                
                with strat_col1:
                    st.markdown("#### ⚡ Aggressive Setup (95% Prob)")
                    if adv.get('Aggressive'):
                        sig = adv['Aggressive']
                        st.success(f"**Strategy**: {sig['Type']}")
                        st.write(f"**Logic**: {sig['Reason']}")
                        st.code(f"Sell Strike: ${sig.get('Strike_Short', 0):.2f}\nBuy Strike:  ${sig.get('Strike_Long', 0):.2f}")
                    else:
                        st.caption("No Aggressive Signal (Conditions not met)")

                with strat_col2:
                    st.markdown("#### 🛡️ Conservative Setup (99.7% Prob)")
                    if adv.get('Conservative'):
                        sig = adv['Conservative']
                        st.success(f"**Strategy**: {sig['Type']}")
                        st.write(f"**Logic**: {sig['Reason']}")
                        st.code(f"Sell Strike: ${sig.get('Strike_Short', 0):.2f}\nBuy Strike:  ${sig.get('Strike_Long', 0):.2f}")
                    else:
                        st.caption("No Conservative Signal (Conditions not met)")

            # --- Dynamic Position Sizing & Risk Management ---
            st.markdown("---")
            st.markdown("#### 💰 Portfolio & Risk Management")
            
            # Logic: 2% Risk Rule
            risk_per_trade_pct = 0.05 # Aggressive 5% for small accounts, or make configurable
            max_risk = account_value * risk_per_trade_pct
            
            # Assumptions (Standardized for Scanner)
            # In a real app, this would come from the specific Option Contract margin
            spread_width = 5.0 
            max_loss_per_contract = spread_width * 100 # $500 margin per spread
            
            # Conservative Yield Estimate (Credit ~ 1/3 of width for aggressive, 1/5 for conservative)
            # We'll use 20% of width as a safe baseline for "High Prob" setups
            estimated_credit = max_loss_per_contract * 0.20 # $100
            
            # Calculate Sizing
            rec_contracts = max(1, int(max_risk / max_loss_per_contract))
            capital_required = rec_contracts * max_loss_per_contract
            proj_profit_trade = rec_contracts * estimated_credit
            
            # Annualized Return Calculation
            holding_days = 30 # Avg DTE
            trade_yield = (estimated_credit / max_loss_per_contract)
            annualized_yield = trade_yield * (365 / holding_days) * 100
            
            # Display Sizing Metrics
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            p_col1.metric("Rec. Size", f"{rec_contracts} Contracts", f"Risk: ${max_risk:.0f}")
            p_col2.metric("Capital Req.", f"${capital_required:.0f}", f"{((capital_required/account_value)*100):.1f}% of Portfolio")
            p_col3.metric("Est. Profit", f"${proj_profit_trade:.0f}", f"Yield: {trade_yield:.1%}")
            
            # Target Return Validation
            delta_color = "normal" if annualized_yield >= target_return else "inverse"
            icon = "✅" if annualized_yield >= target_return else "⚠️"
            p_col4.metric("Ann. Return", f"{annualized_yield:.1f}%", f"Target: {target_return}%", delta_color=delta_color)
            
            if annualized_yield < target_return:
                st.caption(f"{icon} **Yield Warning**: This trade projects {annualized_yield:.1f}% annualized, which is below your {target_return}% target. Consider wider spreads or closer strikes.")

else:
    # Empty State
    st.info("👆 Click 'Run Live Scan' to start the analysis engine.")

# --- Backtest Section ---
# --- Backtest Section ---
with tab_backtest:
    st.header("⏳ Hedge Fund Simulator (Portfolio Mode)")
    st.caption("Simulate Multi-Asset Portfolio with Dynamic Compounding & Profit Taking")
    
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        # Multi-Select for Portfolio
        bt_tickers = st.multiselect("Select Assets", settings.WATCHLIST, default=["SPY", "QQQ", "IWM", "TSLA", "NVDA"])
    with b_col2:
        bt_capital = st.number_input("Start Capital", value=10000, step=1000)
    with b_col3:
        # Default to Dec 2024
        default_start = datetime(2024, 12, 1)
        start_date = st.date_input("Start Date", default_start)
    with b_col4:
        # Default to Dec 2025
        default_end = datetime(2025, 12, 31)
        end_date = st.date_input("End Date", default_end)
    
    if st.button("Run Hedge Fund Simulation", use_container_width=True):
        from backend.app.strategies.backtester import Backtester
        from backend.app.services.yfinance_service import yfinance_service
        
        if not bt_tickers:
            st.error("Please select at least one asset.")
        else:
            with st.spinner(f"Simulating trades from {start_date} to {end_date}..."):
                # Fetch Data Map
                data_map = {}
                progress_bar = st.progress(0)
                
                # Fetch sufficiently long history (needs warmup before start_date)
                # 800 days covers back to late 2023, providing plenty of warmup for Dec 2024 start.
                
                for idx, ticker in enumerate(bt_tickers):
                    # Fetch History
                    data_map[ticker] = yfinance_service.get_historical_data(ticker, days=800)
                    progress_bar.progress((idx + 1) / len(bt_tickers))
                
                # Init Engine
                bt = Backtester(initial_capital=bt_capital, max_positions=5)
                
                # Run Portfolio Sim
                result = bt.run_portfolio(data_map, start_date=start_date, end_date=end_date)
                
                if "error" in result:
                    st.error(result['error'])
                else:
                    stats = result['stats']
                    trades = result['trades']
                    equity = result['equity']
                    
                    # Results UI
                    st.success("Simulation Complete!")
                    
                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    roi_color = "normal" if stats['Total Return'] > 0 else "inverse"
                    m1.metric("Final Capital", f"${stats['Final Capital']:.0f}", f"{(stats['Total Return']*100):.1f}%", delta_color=roi_color)
                    m2.metric("Win Rate", f"{(stats['Win Rate']*100):.1f}%")
                    m3.metric("Total Trades", stats['Trades'])
                    
                    # CAGR Calculation
                    days_simulated = 252 # approx
                    final_eq = stats['Final Capital']
                    cagr = ((final_eq / bt_capital) ** (252/days_simulated)) - 1
                    m4.metric("CAGR (Est.)", f"{cagr:.1%}")
                    
                    # Row 2: Advanced Risk Metrics
                    st.markdown("##### 🛡️ Risk & Performance")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Max Drawdown", f"{stats.get('Max Drawdown', 0):.1%}", delta_color="inverse")
                    r2.metric("Profit Factor", f"{stats.get('Profit Factor', 0):.2f}x")
                    r3.metric("Sharpe Ratio", f"{stats.get('Sharpe Ratio', 0):.2f}")
                    r4.metric("Expectancy", f"${stats.get('Expectancy', 0):.0f}/trade")
                    
                    # Plot
                    if not equity.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=equity['date'], y=equity['equity'], fill='tozeroy', name='Equity', line=dict(color='#3b82f6')))
                        fig.add_trace(go.Scatter(x=equity['date'], y=equity['cash'], name='Cash', line=dict(color='#10b981', dash='dash')))
                        fig.update_layout(title="Portfolio Growth (Compound)", template="plotly_dark", plot_bgcolor="#1e293b", margin=dict(l=20,r=20,t=40,b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade Log
                    st.subheader("📜 Recent Trades")
                    if not trades.empty:
                         trades['entry'] = pd.to_datetime(trades['entry']).dt.strftime('%Y-%m-%d')
                         
                         # Color code PnL
                         def highlight_pnl(val):
                             color = '#10b981' if val > 0 else '#ef4444'
                             return f'color: {color}'
                         
                         # Rename for UI
                         trades_ui = trades.rename(columns={
                             "ticker": "Ticker",
                             "contracts": "Size",
                             "entry": "Entry",
                             "type": "Strategy",
                             "strikes": "Stike(s)",
                             "reason": "Comment",
                             "pnl": "P&L",
                             "return_pct": "Returns %"
                         })
                         
                         st.dataframe(
                             trades_ui.style.map(highlight_pnl, subset=['P&L', 'Returns %'])
                                   .format({'P&L': "${:.0f}", 'Returns %': "{:.1%}"}),
                             use_container_width=True, 
                             hide_index=True
                         )
                    else:
                        st.warning("No trades triggered. Market might be too choppy for this strict logic.")

