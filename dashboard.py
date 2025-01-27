__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import yfinance as yf
from market_analysis_crew import MarketAnalysisCrew

# Set page configuration
st.set_page_config(
    page_title="Elite Stock Analysis AI",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    div.block-container {
        padding-top: 2rem;
    }
    div.element-container {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def create_price_chart(ticker_data):
    """Create a price chart using Plotly"""
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1, 2, 3],  # Placeholder x values
            y=[ticker_data["price_history"]["last_month_low"], 
               ticker_data["current_price"], 
               ticker_data["price_history"]["last_month_high"]],
            mode='lines+markers',
            name='Price'
        ))
        fig.update_layout(
            title=f'Price Range - {ticker_data["company_name"]}',
            yaxis_title='Price ($)',
            showlegend=True
        )
        return fig
    except Exception as e:
        st.error(f"Error creating price chart: {str(e)}")
        return None

def create_volume_chart(ticker_data):
    """Create a volume chart using Plotly"""
    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Average Daily Volume'],
            y=[ticker_data["price_history"]["last_month_avg_volume"]],
            name='Volume'
        ))
        fig.update_layout(
            title=f'Trading Volume - {ticker_data["company_name"]}',
            yaxis_title='Volume',
            showlegend=True
        )
        return fig
    except Exception as e:
        st.error(f"Error creating volume chart: {str(e)}")
        return None

def display_metrics(ticker_data):
    """Display key metrics in columns"""
    try:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${ticker_data['current_price']:.2f}")
        with col2:
            st.metric("P/E Ratio", f"{ticker_data['pe_ratio']:.2f}")
        with col3:
            st.metric("Market Cap", f"${ticker_data['market_cap']/1e9:.2f}B")
        with col4:
            st.metric("Dividend Yield", f"{ticker_data['dividend_yield']*100:.2f}%")
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

def display_analysis_section(title, data):
    """Display an analysis section with proper error handling"""
    try:
        st.subheader(title)
        if isinstance(data, dict) and data:
            st.json(data)
        elif isinstance(data, str):
            st.write(data)
        else:
            st.warning(f"No {title.lower()} data available")
    except Exception as e:
        st.error(f"Error displaying {title.lower()}: {str(e)}")

def format_json_output(data):
    """Format JSON data for better display"""
    try:
        if isinstance(data, str):
            # Try to parse if it's a string
            data = json.loads(data)
        return json.dumps(data, indent=2)
    except:
        return data

def display_agent_analysis(title, data, icon):
    """Display an agent's analysis with better formatting"""
    st.markdown(f"### {icon} {title}")
    
    if not data:
        st.info(f"No {title.lower()} data available")
        return
        
    # Create columns for different aspects of the analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Analysis Results")
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    with st.expander(key.replace('_', ' ').title()):
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                st.markdown(f"**{sub_key.replace('_', ' ').title()}:**")
                                st.write(sub_value)
                        elif isinstance(value, list):
                            for item in value:
                                st.write(f"- {item}")
                        else:
                            st.write(value)
            else:
                st.write(data)
        except Exception as e:
            st.error(f"Error displaying data: {str(e)}")
    
    with col2:
        st.markdown("#### Key Takeaways")
        try:
            if isinstance(data, dict):
                # Extract and display key metrics based on analysis type
                if title == "Market Research":
                    if "market_sentiment" in data:
                        st.metric("Market Sentiment", data["market_sentiment"])
                elif title == "Technical Analysis":
                    if "price_targets" in data and "target_price" in data["price_targets"]:
                        st.metric("Price Target", data["price_targets"]["target_price"])
                elif title == "Fundamental Analysis":
                    if "valuation" in data and "fair_value" in data["valuation"]:
                        st.metric("Fair Value", data["valuation"]["fair_value"])
                elif title == "Investment Strategy":
                    if "recommendation" in data and "action" in data["recommendation"]:
                        st.metric("Recommendation", data["recommendation"]["action"])
        except Exception as e:
            st.warning("Could not display key metrics")

def main():
    # Header with logo and title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/stocks.png", width=80)
    with col2:
        st.title("Elite Stock Analysis AI")
        st.markdown("*Powered by AI Agents for Comprehensive Stock Analysis*")
    
    # Input section with better styling
    st.markdown("---")
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()
        with col2:
            st.write("")
            st.write("")
            analyze_button = st.button("🔍 Analyze Stock", use_container_width=True)
    
    if analyze_button:
        with st.spinner("🤖 AI Agents are analyzing your stock..."):
            try:
                market_crew = MarketAnalysisCrew()
                analysis = market_crew.analyze_stock(ticker)
                
                if "error" in analysis and analysis["error"]:
                    st.error(analysis["error"])
                else:
                    # Display analysis results in tabs
                    if analysis.get("json_output"):
                        data = analysis["json_output"]
                        tabs = st.tabs([
                            "📊 Market Research",
                            "📈 Technical Analysis",
                            "💰 Fundamental Analysis",
                            "🎯 Investment Strategy",
                            "🔍 Raw Data"
                        ])
                        
                        with tabs[0]:
                            display_agent_analysis(
                                "Market Research",
                                data[0] if isinstance(data, list) and len(data) > 0 else {},
                                "🔎"
                            )
                        
                        with tabs[1]:
                            display_agent_analysis(
                                "Technical Analysis",
                                data[1] if isinstance(data, list) and len(data) > 1 else {},
                                "📊"
                            )
                        
                        with tabs[2]:
                            display_agent_analysis(
                                "Fundamental Analysis",
                                data[2] if isinstance(data, list) and len(data) > 2 else {},
                                "📈"
                            )
                        
                        with tabs[3]:
                            display_agent_analysis(
                                "Investment Strategy",
                                data[3] if isinstance(data, list) and len(data) > 3 else {},
                                "🎯"
                            )
                        
                        with tabs[4]:
                            st.markdown("### 🔍 Raw Analysis Data")
                            with st.expander("View Raw JSON Output"):
                                st.code(format_json_output(analysis["json_output"]))
                            if analysis.get("token_usage"):
                                with st.expander("View Token Usage"):
                                    st.json(analysis["token_usage"])
                    
                    # Display agent reasoning if available
                    if analysis.get("tasks_output"):
                        st.markdown("---")
                        st.markdown("### 🤖 Agent Reasoning")
                        for i, task in enumerate(analysis["tasks_output"]):
                            with st.expander(f"View Agent {i+1} Reasoning Process"):
                                st.write(task)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using CrewAI and Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 