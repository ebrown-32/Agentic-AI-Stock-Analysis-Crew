__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from market_analysis_crew import MarketAnalysisCrew, LLM_PROVIDERS, get_available_providers
from datetime import datetime
import numpy as np
import plotly.express as px
import json
import time
from typing import Any
import os

st.set_page_config(
    page_title="AI Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

INVESTMENT_TERMS = {
    "RSI": "Relative Strength Index - A momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions.",
    "MACD": "Moving Average Convergence Divergence - A trend-following momentum indicator that shows the relationship between two moving averages of a security's price.",
    "Moving Average": "A calculation used to analyze data points by creating a series of averages of different subsets of the full data set.",
    "Volume Profile": "A visualization of trading activity over a specified period that shows the price levels where the most trading activity occurred.",
    "Beta": "A measure of a stock's volatility in relation to the overall market.",
    "Value at Risk (VaR)": "A statistical measure of the potential loss in value of a portfolio over a defined period.",
    "Sharpe Ratio": "A measure that indicates the average return minus the risk-free return divided by the standard deviation of return on an investment.",
    "DCF Valuation": "Discounted Cash Flow - A valuation method that estimates the value of an investment based on its expected future cash flows.",
    "P/E Ratio": "Price-to-Earnings Ratio - A valuation measure that compares a company's stock price to its earnings per share.",
    "Market Cap": "The total value of a company's shares of stock, calculated by multiplying the price of a stock by its total number of outstanding shares.",
    "Economic Moat": "A company's competitive advantage that allows it to maintain profitability and market share over time.",
    "ESG": "Environmental, Social, and Governance - A set of standards for a company's operations that socially conscious investors use to screen potential investments."
}

st.markdown("""
    <style>
        /* Main container styling */
        .stApp {
            max-width: 100%;
            margin: 0;
            padding: 0;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            width: 300px !important;
            background-color: #f8f9fa;
            padding: 2rem;
            position: fixed;
            left: 0;
            height: 100%;
            border-right: 1px solid #e9ecef;
        }

        /* Main content area styling */
        section[data-testid="stMainContent"] {
            margin-left: 300px;
            padding: 2rem;
            max-width: calc(100% - 300px);
        }

        /* Card styling */
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }

        .analysis-section {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        }

        /* Headers and text styling */
        .header-style {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #1f77b4;
        }

        .analysis-header {
            color: #1f77b4;
            font-size: 1.2em;
            margin-bottom: 10px;
        }

        /* Tooltip styling */
        .term-tooltip {
            text-decoration: underline dotted;
            cursor: help;
        }

        /* Code output styling */
        .json-output {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            white-space: pre-wrap;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Ensure content doesn't overlap with sidebar on smaller screens */
        @media (max-width: 768px) {
            section[data-testid="stMainContent"] {
                margin-left: 0;
                max-width: 100%;
            }
        }

        /* Agent Chat styling */
        .agent-chat {
            max-height: 600px;
            overflow-y: auto;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid #e9ecef;
        }
        
        .agent-message {
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #1f77b4;
            animation: fadeIn 0.5s ease-in;
        }
        
        .agent-name {
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .agent-timestamp {
            font-size: 0.8em;
            color: #666;
        }
        
        .agent-thinking {
            color: #666;
            font-style: italic;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .thinking-dots {
            display: inline-block;
            animation: thinking 1.5s infinite;
        }
        
        .agent-result {
            margin-top: 10px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes thinking {
            0% { content: '.'; }
            33% { content: '..'; }
            66% { content: '...'; }
        }

        /* Tooltip styling */
        .sidebar-tooltip {
            color: #1f77b4;
            font-size: 0.8em;
            margin-top: 5px;
            padding: 5px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }
    </style>
""", unsafe_allow_html=True)

def format_json_output(data: dict) -> str:
    """Format JSON data for better readability"""
    return json.dumps(data, indent=2)

def add_tooltips_to_text(text: str) -> str:
    """Add tooltips to technical terms in the text"""
    for term, definition in INVESTMENT_TERMS.items():
        if term in text:
            text = text.replace(term, f'<span class="term-tooltip" title="{definition}">{term}</span>')
    return text

def create_candlestick_chart(stock_data):
    """Create an interactive candlestick chart with volume"""
    dates = pd.to_datetime(stock_data["dates"])
    prices = stock_data["price_history"]
    volumes = stock_data["volume_history"]
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
                       row_heights=[0.7, 0.3])

    # Add candlestick
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='#1f77b4')
    ), row=1, col=1)

    # Add volume bar chart
    fig.add_trace(go.Bar(
        x=dates,
        y=volumes,
        name='Volume',
        marker_color='#2ca02c'
    ), row=2, col=1)

    # Add moving averages
    ma_data = stock_data["technical_indicators"]["moving_averages"]
    for ma_name, ma_values in ma_data.items():
        fig.add_trace(go.Scatter(
            x=dates,
            y=ma_values,
            name=ma_name,
            line=dict(dash='dash')
        ), row=1, col=1)

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Price and Volume Analysis",
        xaxis_rangeslider_visible=False
    )

    return fig

def create_technical_indicators_chart(stock_data):
    """Create technical indicators visualization"""
    dates = pd.to_datetime(stock_data["dates"])
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05, subplot_titles=('RSI', 'MACD'))

    # Add RSI
    rsi_values = stock_data["technical_indicators"]["rsi"]
    fig.add_trace(go.Scatter(
        x=dates,
        y=rsi_values,
        name='RSI',
        line=dict(color='#1f77b4')
    ), row=1, col=1)

    # Add MACD
    macd_data = stock_data["technical_indicators"]["macd"]
    fig.add_trace(go.Scatter(
        x=dates,
        y=macd_data["macd"],
        name='MACD',
        line=dict(color='#1f77b4')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=macd_data["signal"],
        name='Signal',
        line=dict(color='#ff7f0e')
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Technical Indicators"
    )

    # Add RSI lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

    return fig

def display_metrics_dashboard(metrics):
    """Display financial metrics in an organized dashboard"""
    cols = st.columns(3)
    
    # Profitability Metrics
    with cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📈 Profitability")
        for key, value in metrics["profitability"].items():
            if value is not None:
                st.metric(
                    label=key.replace('_', ' ').title(),
                    value=f"{value:.2%}" if isinstance(value, float) else value
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # Valuation Metrics
    with cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("💰 Valuation")
        for key, value in metrics["valuation"].items():
            if value is not None:
                st.metric(
                    label=key.replace('_', ' ').title(),
                    value=f"{value:.2f}" if isinstance(value, float) else value
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # Growth Metrics
    with cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("🚀 Growth")
        for key, value in metrics["growth"].items():
            if value is not None:
                st.metric(
                    label=key.replace('_', ' ').title(),
                    value=f"{value:.2%}" if isinstance(value, float) else value
                )
        st.markdown('</div>', unsafe_allow_html=True)

def display_risk_metrics(risk_metrics):
    """Display risk metrics with visual indicators"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("### 🎯 Risk Analysis")
    
    cols = st.columns(4)
    
    # Volatility
    with cols[0]:
        volatility = risk_metrics.get("volatility")
        if volatility:
            st.metric(
                "Annualized Volatility",
                f"{volatility:.2%}",
                delta_color="inverse"
            )

    # Value at Risk
    with cols[1]:
        var = risk_metrics.get("value_at_risk")
        if var:
            st.metric(
                "Daily VaR (95%)",
                f"{var:.2%}",
                delta_color="inverse"
            )

    # Sharpe Ratio
    with cols[2]:
        sharpe = risk_metrics.get("sharpe_ratio")
        if sharpe:
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta_color="normal"
            )

    # Risk Assessment
    with cols[3]:
        risk_level = risk_metrics.get("risk_assessment", "").upper()
        if risk_level:
            color = {
                "LOW": "green",
                "MEDIUM": "orange",
                "HIGH": "red"
            }.get(risk_level, "gray")
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h4>Risk Level</h4>
                    <p style='color: {color}; font-size: 20px; font-weight: bold;'>
                        {risk_level}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_educational_page():
    """Display educational page with investment terms and definitions"""
    st.markdown("## 📚 Investment Terms Glossary")
    st.markdown("Understanding financial terms is crucial for making informed investment decisions. "
                "Here's a comprehensive glossary of important investment terms used in our analysis.")
    
    cols = st.columns(3)
    terms = list(INVESTMENT_TERMS.items())
    terms_per_col = len(terms) // 3 + (len(terms) % 3 > 0)
    
    for i, col in enumerate(cols):
        with col:
            start_idx = i * terms_per_col
            end_idx = min((i + 1) * terms_per_col, len(terms))
            for term, definition in terms[start_idx:end_idx]:
                st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                        <strong>{term}</strong><br>
                        <small>{definition}</small>
                    </div>
                """, unsafe_allow_html=True)

def format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp to readable format"""
    dt = datetime.fromisoformat(timestamp)
    return dt.strftime("%I:%M:%S %p")

def display_agent_message(agent_name: str, message: str, timestamp: str = None, status: str = "progress", result: Any = None):
    """Display a message from an agent in a chat-like format"""
    timestamp_str = format_timestamp(timestamp) if timestamp else datetime.now().strftime("%I:%M:%S %p")
    
    st.markdown(f"""
        <div class="agent-message">
            <div class="agent-name">
                <span>{agent_name}</span>
                <span class="agent-timestamp">{timestamp_str}</span>
            </div>
            <div class="{'agent-thinking' if status == 'progress' else ''}">
                {message}
                {' <span class="thinking-dots">...</span>' if status == 'progress' else ''}
            </div>
            {f'<div class="agent-result">{result}</div>' if result else ''}
        </div>
    """, unsafe_allow_html=True)

def display_agent_chat(crew: MarketAnalysisCrew):
    """Display the agent chat with real-time updates"""
    chat_placeholder = st.empty()
    
    with chat_placeholder.container():
        st.markdown('<div class="agent-chat">', unsafe_allow_html=True)
        
        # Initialize session state for messages if not exists
        if 'agent_messages' not in st.session_state:
            st.session_state.agent_messages = []
        
        # Get new messages
        new_messages = crew.get_agent_messages()
        if new_messages:
            st.session_state.agent_messages.extend(new_messages)
        
        # Display all messages
        for msg in st.session_state.agent_messages:
            display_agent_message(
                agent_name=msg["agent"],
                message=msg["message"],
                timestamp=msg["timestamp"],
                status=msg["status"],
                result=msg["result"] if msg["status"] == "complete" else None
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Auto-scroll to bottom (using JavaScript)
        st.markdown("""
            <script>
                var chat = document.querySelector('.agent-chat');
                chat.scrollTop = chat.scrollHeight;
            </script>
        """, unsafe_allow_html=True)

def display_ai_analysis(results: dict):
    """Display AI analysis results in a structured format"""
    if not results:
        return

    if isinstance(results, dict) and "raw_analysis" in results:
        st.markdown(add_tooltips_to_text(results["raw_analysis"]), unsafe_allow_html=True)
        return

    # Map analysis keys to their display names
    analysis_mapping = {
        "market_research": "Market Research",
        "technical_analysis": "Technical Analysis",
        "fundamental_analysis": "Fundamental Analysis",
        "risk_analysis": "Risk Analysis",
        "investment_strategy": "Investment Strategy"
    }

    for key, content in results.items():
        if key == "risk_metrics":
            continue
            
        display_name = analysis_mapping.get(key, key.replace('_', ' ').title())
        st.markdown(f"<h3 class='analysis-header'>{display_name}</h3>", unsafe_allow_html=True)
        
        if isinstance(content, dict):
            formatted_content = format_json_output(content)
            formatted_content_with_tooltips = add_tooltips_to_text(formatted_content)
            
            st.markdown("<div class='json-output'>", unsafe_allow_html=True)
            st.markdown(f"```json\n{formatted_content}\n```")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if key == "investment_strategy":
                st.markdown("#### 📝 Key Points:")
                st.markdown("- **Investment Thesis**: The core reasoning behind the investment recommendation")
                st.markdown("- **Position Strategy**: Guidelines for position sizing and portfolio allocation")
                st.markdown("- **Execution Plan**: Specific entry and exit points with timing considerations")
                st.markdown("- **Risk Management**: Position limits and monitoring requirements")
        else:
            st.markdown(add_tooltips_to_text(str(content)), unsafe_allow_html=True)

def main():
    # Sidebar
    with st.sidebar:
        # Navigation at the top
        st.markdown("### Navigation")
        page = st.radio(
            "Go to:",
            ["Dashboard", "Educational Resources"]
        )
        
        st.markdown("---")
        
        st.markdown("### Stock Analysis Settings")
        ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()
        
        # LLM Provider Selection
        st.markdown("##### AI Model Configuration")
        available_providers = get_available_providers()
        
        if not available_providers:
            st.warning("⚠️ No LLM providers configured!")
            st.markdown("""
                <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                    <strong>Quick Setup:</strong><br>
                    1. Copy <code>.env.example</code> to <code>.env</code><br>
                    2. Add your API key to <code>.env</code><br>
                    3. Restart the application<br><br>
                    <strong>Free Options:</strong><br>
                    • <a href="https://platform.openai.com/signup">OpenAI</a> - $5 free credits<br>
                    • <a href="https://console.anthropic.com/">Anthropic</a> - Free tier available<br>
                    • <a href="https://makersuite.google.com/app/apikey">Google AI</a> - Free API key<br>
                    • <a href="https://ollama.ai/">Ollama</a> - Free local models
                </div>
            """, unsafe_allow_html=True)
        
        # Provider selection
        provider_options = []
        model_options = {}
        
        for provider in available_providers:
            provider_config = LLM_PROVIDERS[provider]
            for model in provider_config["models"]:
                display_name = f"{provider.upper()} - {model}"
                provider_key = f"{provider}/{model}"
                provider_options.append(display_name)
                model_options[display_name] = provider_key
        
        if provider_options:
            selected_display = st.selectbox(
                "Select AI Model:",
                provider_options,
                index=0,
                help="Choose the AI model for analysis. Different models have different strengths and costs."
            )
            selected_model = model_options[selected_display]
            
            # Show model info
            provider_name = selected_model.split('/')[0]
            st.markdown(f"""
                <div class="sidebar-tooltip">
                    🤖 Using {provider_name.upper()} model. Each provider offers different analysis styles and capabilities.
                </div>
            """, unsafe_allow_html=True)
        else:
            selected_model = None
            
        st.markdown("##### Time Period")
        time_period = st.selectbox(
            "Select Time Period:",
            ["1y", "6mo", "3mo", "1mo"],
            index=0,
            help="Select how far back the analysis should go: 1 year (1y), 6 months (6mo), 3 months (3mo), or 1 month (1mo). This affects the historical data used in the analysis."
        )
        st.markdown("""
            <div class="sidebar-tooltip">
                📅 This determines the timeframe of historical data used in the analysis, affecting metrics like trends, moving averages, and volatility calculations.
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Analyze Stock", type="primary"):
            if not available_providers:
                st.error("❌ **No LLM providers available**\n\nPlease configure at least one API key in your .env file.")
                st.stop()
                
            with st.spinner("Initializing analysis..."):
                try:
                    crew = MarketAnalysisCrew(model_provider=selected_model)
                    
                    # progress placeholder
                    progress_placeholder = st.empty()
                    
                    with progress_placeholder.container():
                        # Initialize analysis
                        analysis_results = crew.analyze_stock(ticker)
                        
                        # Check if analysis returned an error
                        if isinstance(analysis_results, dict) and "error" in analysis_results:
                            reason = analysis_results.get("reason", "unknown")
                            if reason == "missing_api_key":
                                st.error("🔑 **API Key Missing**\n\nPlease set your API key as an environment variable:")
                                st.code("export GOOGLE_API_KEY=your_api_key_here")
                                st.info("💡 **Tip**: You can get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
                            elif reason == "configuration_error":
                                st.error("⚙️ **Configuration Error**\n\n" + analysis_results["error"])
                            else:
                                st.error("❌ **Analysis Failed**\n\n" + analysis_results["error"])
                            return
                        
                        st.session_state.analysis_results = analysis_results
                        st.session_state.stock_data = crew.tools["stock_data"]._run(ticker)
                        st.session_state.financial_metrics = crew.tools["financial_metrics"]._run(ticker)
                        
                        # Display real-time agent chat
                        display_agent_chat(crew)
                        
                        # Update every second while analysis is running
                        while not hasattr(st.session_state, 'analysis_results'):
                            display_agent_chat(crew)
                            time.sleep(1)
                    
                except ValueError as e:
                    if "API key" in str(e).lower():
                        st.error("🔑 **API Key Missing**\n\nPlease set your API key as an environment variable:")
                        st.code("export GOOGLE_API_KEY=your_api_key_here")
                        st.info("💡 **Tip**: You can get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
                    else:
                        st.error(f"⚙️ **Configuration Error**: {str(e)}")
                    return
                except Exception as e:
                    st.error(f"❌ **Unexpected Error**: {str(e)}")
                    st.info("Please check your configuration and try again.")
                    return

    if page == "Educational Resources":
        display_educational_page()
        return

    # Main dashboard content
    st.title("AI Stock Analysis Dashboard")
    st.markdown("---")

    if hasattr(st.session_state, 'analysis_results'):
        # Stock Overview Section
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        cols = st.columns(4)
        stock_data = st.session_state.stock_data
        
        with cols[0]:
            st.metric("Current Price", f"${stock_data['current_price']:.2f}")
        with cols[1]:
            st.metric("Market Cap", f"${stock_data['market_cap']:,.0f}")
        with cols[2]:
            st.metric("52-Week High", f"${stock_data['52_week_high']:.2f}")
        with cols[3]:
            st.metric("52-Week Low", f"${stock_data['52_week_low']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Charts Section
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📈 Price Analysis", "📊 Technical Indicators"])
        
        with tab1:
            st.plotly_chart(create_candlestick_chart(stock_data), use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_technical_indicators_chart(stock_data), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Financial Metrics Section
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("### �� Financial Analysis")
        display_metrics_dashboard(st.session_state.financial_metrics)
        st.markdown('</div>', unsafe_allow_html=True)

        # Risk Metrics Section
        display_risk_metrics(st.session_state.analysis_results.get("risk_metrics", {}))

        # AI Analysis Results Section
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("### 🤖 AI Analysis Insights")
        display_ai_analysis(st.session_state.analysis_results)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👈 Enter a stock ticker in the sidebar and click 'Analyze Stock' to begin analysis.")

if __name__ == "__main__":
    main() 