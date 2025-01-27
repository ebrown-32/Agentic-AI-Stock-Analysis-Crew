# Elite Stock Analysis AI

A sophisticated stock analysis system powered by CrewAI and Hugging Face models, providing comprehensive market insights through a team of specialized AI agents.

## Features

- 🤖 **AI-Powered Analysis**: Team of specialized AI agents working together
  - Elite Market Researcher
  - Senior Technical Analyst
  - Chief Fundamental Analyst
  - Sentiment Analysis Specialist
  - Portfolio Strategy Expert

- 📊 **Comprehensive Analysis**:
  - Market research and sector analysis
  - Multi-timeframe technical analysis
  - Deep fundamental analysis
  - Sentiment analysis from multiple sources
  - Competitor analysis
  - Institutional holdings tracking
  - Risk assessment and portfolio strategy

- 💼 **Investment Insights**:
  - Actionable investment recommendations
  - Risk management guidelines
  - Entry and exit points
  - Portfolio fit analysis
  - Market sentiment evaluation

- 🎯 **User-Focused Features**:
  - Customizable analysis timeframes
  - Risk tolerance preferences
  - Interactive web interface
  - Detailed analysis reports

## Prerequisites

- Python 3.8 or higher
- Hugging Face account (for model access)
- Internet connection for real-time market data

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd elite-stock-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```env
HUGGINGFACE_API_KEY=your_api_key_here
```

## Configuration

1. **Model Selection**:
   - The default model is set to "meta-llama/Llama-2-70b-chat-hf"
   - You can modify the `MODEL_NAME` in `main.py` to use a different model

2. **Additional APIs** (Optional):
   - Set up API keys for enhanced features:
     - Alpha Vantage
     - Finnhub
     - Other financial data providers

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Access the web interface:
   - Open your browser and go to `http://localhost:8501`
   - The interface will be displayed with the following sections:
     - Query input
     - Timeframe selection
     - Risk tolerance adjustment
     - Analysis results

3. Enter your analysis query:
   - Type your investment question
   - Select analysis timeframe
   - Adjust risk tolerance
   - Click "Generate Elite Analysis"

4. Review the analysis:
   - The system will process your request through multiple specialized agents
   - A comprehensive report will be generated with actionable insights

## Example Queries

- "Analyze AAPL for a long-term value investment portfolio, considering current market conditions and risks."
- "Compare TSLA and its competitors in the EV market for short-term trading opportunities."
- "Evaluate MSFT's potential as a defensive stock during market uncertainty."

## Understanding the Results

The analysis report includes:
1. **Executive Summary**
   - Key findings and recommendations

2. **Market Analysis**
   - Industry trends
   - Competitive position
   - Market conditions

3. **Technical Analysis**
   - Price trends
   - Support/resistance levels
   - Volume patterns
   - Momentum indicators

4. **Fundamental Analysis**
   - Financial metrics
   - Company health
   - Growth prospects
   - Risk factors

5. **Sentiment Analysis**
   - News sentiment
   - Social media trends
   - Analyst recommendations

6. **Investment Strategy**
   - Specific recommendations
   - Risk management
   - Position sizing
   - Portfolio considerations

## Customization

You can customize the analysis by:
1. Modifying agent configurations in `main.py`
2. Adjusting analysis parameters
3. Adding new tools and data sources
4. Customizing the report format

## Troubleshooting

Common issues and solutions:

1. **Model Loading Errors**:
   - Verify Hugging Face API key
   - Check model availability
   - Ensure sufficient system resources

2. **Data Fetching Issues**:
   - Check internet connection
   - Verify API keys
   - Ensure valid stock symbols

3. **Performance Issues**:
   - Reduce analysis scope
   - Upgrade system resources
   - Check for resource conflicts

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CrewAI framework
- Hugging Face models
- yfinance library
- Streamlit framework

## Support

For support:
- Create an issue in the repository
- Contact the maintainers
- Check the documentation

## Disclaimer

This tool is for informational purposes only. Do not make investment decisions solely based on this analysis. Always conduct your own research and consult with financial advisors. 