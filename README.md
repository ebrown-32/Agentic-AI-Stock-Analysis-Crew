# Stock Analyst AI Team

This is an exploratory project to get familiar with CrewAI, an orchestration framework for agentic AI. It is built in pure python with a Streamlit frontend. The idea is to provide a free, intelligent stock analysis platform powered by a team of AI agents specializing in different aspects of market analysis. Investing should be accessible to everyone.
(This tool is for informational purposes only. Always conduct your own research and consult with financial advisors before making investment decisions. Feel free to extend this tool to your own needs.)

## Status: Work in Progress 🚧

I'm making things better as I have the time! Some stuff is broken, but I'm working on it.

## Overview

The Stock Analyst AI Team combines multiple specialized AI agents to provide comprehensive stock analysis:
- Market Intelligence Officer
- Technical Analysis Specialist
- Fundamental Analysis Expert
- Portfolio Strategy Expert

Currently the agents are equipped with the tools to 1. use the Yahoo Finance API to get market data. 2. Search the web for news and other information. 3. Use Google Gemini (https://gemini.google.com/) as the LLM.

## Features

✅ **Current Features**
- Real-time market data visualization
- Interactive candlestick and volume charts
- Key financial metrics display
- AI-powered market analysis
- Technical and fundamental analysis
- Investment recommendations

🚧 **Roadmap**
TBD

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your environment variables in `.env`:
```
GEMINI_API_KEY=your_api_key_here
MODEL_PROVIDER=gemini/gemini-1.5-pro-latest
```

## Usage

Run the dashboard:
```bash
streamlit run dashboard.py
```

Enter a stock ticker and click "Analyze Stock" to receive:
- Market data visualization
- Technical analysis
- Fundamental analysis
- Investment recommendations

## Project Structure

```
stock-analyst-ai/
├── market_analysis_crew.py  # AI agent implementation
├── dashboard.py            # Streamlit interface
├── requirements.txt       # Dependencies
└── .env                  # Configuration
```

## Dependencies

- crewai (https://docs.crewai.com/introduction)
- streamlit (https://streamlit.io/)
- yfinance (https://pypi.org/project/yfinance/)
- plotly (https://plotly.com/)
- pandas (https://pandas.pydata.org/)
- python-dotenv (https://pypi.org/project/python-dotenv/)

## Contributing

This is a work in progress, and contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License

## Disclaimer

This tool is for informational purposes only. Always conduct your own research and consult with financial advisors before making investment decisions. 