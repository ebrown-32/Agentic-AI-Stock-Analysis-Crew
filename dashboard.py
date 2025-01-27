__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
from market_analysis_crew import MarketAnalysisCrew

# Set page configuration
st.set_page_config(layout="wide", page_title="Stock Analyst AI Team")

def create_candlestick_chart(ticker_symbol: str, period: str = "1y") -> go.Figure:
    """Create an interactive candlestick chart"""
    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=period)
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        
        fig.update_layout(
            title=f'{ticker_symbol} Stock Price',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating candlestick chart: {str(e)}")
        return None

def create_volume_chart(ticker_symbol: str, period: str = "1y") -> go.Figure:
    """Create an interactive volume chart"""
    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=period)
        
        fig = go.Figure(data=[go.Bar(x=df.index, y=df['Volume'], name='Volume')])
        
        fig.update_layout(
            title=f'{ticker_symbol} Trading Volume',
            yaxis_title='Volume',
            template='plotly_dark'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating volume chart: {str(e)}")
        return None

def display_key_metrics(ticker_symbol: str) -> None:
    """Display key financial metrics in a modern layout"""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${info.get('currentPrice', 'N/A'):,.2f}",
                f"{info.get('regularMarketChangePercent', 0):.2f}%"
            )
        
        with col2:
            st.metric(
                "Market Cap",
                f"${info.get('marketCap', 0)/1e9:,.2f}B",
                "Market Value"
            )
        
        with col3:
            st.metric(
                "P/E Ratio",
                f"{info.get('trailingPE', 'N/A')}"
            )
        
        with col4:
            st.metric(
                "52W Range",
                f"${info.get('fiftyTwoWeekLow', 0):,.2f} - ${info.get('fiftyTwoWeekHigh', 0):,.2f}"
            )
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

def display_analysis_section(title: str, data: dict) -> None:
    """Display analysis results in a modern, organized layout"""
    try:
        st.subheader(title)
        
        if isinstance(data, dict):
            for key, value in data.items():
                with st.expander(f"📊 {key.replace('_', ' ').title()}"):
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            st.markdown(f"**{sub_key.replace('_', ' ').title()}:**")
                            st.write(sub_value)
                    elif isinstance(value, list):
                        for item in value:
                            st.markdown(f"• {item}")
                    else:
                        st.write(value)
        else:
            st.write(data)
    except Exception as e:
        st.error(f"Error displaying analysis section: {str(e)}")

def main():
    # Header with logo and title
    st.markdown("""
        <div style='text-align: center'>
            <h1>🔮 Stock Analyst AI Team</h1>
            <p style='font-size: 1.2em'>Intelligent Stock Analysis Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration only
    with st.sidebar:
        st.markdown("## Analysis Configuration")
        ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()
        period = st.select_slider(
            "Analysis Period:",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            value="1y"
        )
        analyze_button = st.button("🚀 Analyze Stock", use_container_width=True)
    
    # Main content area
    if analyze_button:
        with st.spinner("Analyzing stock... This may take a few minutes..."):
            try:
                # Display key metrics at the top
                display_key_metrics(ticker)
                
                # Create main dashboard layout for charts
                st.markdown("## Market Data Visualization")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    candlestick = create_candlestick_chart(ticker, period)
                    if candlestick:
                        st.plotly_chart(candlestick, use_container_width=True)
                
                with chart_col2:
                    volume = create_volume_chart(ticker, period)
                    if volume:
                        st.plotly_chart(volume, use_container_width=True)
                
                # Perform AI analysis
                market_crew = MarketAnalysisCrew()
                analysis = market_crew.analyze_stock(ticker)
                
                if "error" in analysis and analysis["error"]:
                    st.error(analysis["error"])
                else:
                    st.markdown("## AI Analysis Results")
                    # Create tabs for different types of analysis
                    tabs = st.tabs([
                        "🎯 Investment Strategy",
                        "🔍 Market Research",
                        "📈 Technical Analysis",
                        "📊 Fundamental Analysis",
                        "📑 Raw Data"
                    ])
                    
                    tab_sections = [
                        "investment_strategy",
                        "market_research",
                        "technical_analysis",
                        "fundamental_analysis"
                    ]
                    
                    for i, section in enumerate(tab_sections):
                        with tabs[i]:
                            if analysis.get("json_output"):
                                display_analysis_section(
                                    section.replace("_", " ").title(),
                                    analysis["json_output"].get(section, {})
                                )
                    
                    with tabs[4]:
                        st.json(analysis)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        # Default view when no analysis is running
        st.markdown("""
            ### Welcome to your Stock Analyst AI Team! 👋
            
            Enter a stock ticker in the sidebar and click 'Analyze Stock' to begin your analysis.
            You'll receive:
            
            - 📊 Real-time market data visualization
            - 🤖 AI-powered market analysis
            - 📈 Technical indicators and patterns
            - 🎯 Investment recommendations
            - 📑 Comprehensive research reports
        """)

if __name__ == "__main__":
    main() 