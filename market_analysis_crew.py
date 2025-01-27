from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from financial_tools import stock_data_tool, financial_metrics_tool
import os
from dotenv import load_dotenv
from typing import Any, Dict

# Load environment variables
load_dotenv()

# Initialize LLM with Gemini
llm = LLM(
    model="gemini/gemini-1.5-pro-latest",
    temperature=0.7
)

class MarketAnalysisCrew:
    def __init__(self):
        # Initialize tools with proper input handling
        def search_wrapper(query: str) -> str:
            try:
                search = DuckDuckGoSearchRun()
                return search.run(query)
            except Exception as e:
                return f"Search failed: {str(e)}"
        
        def stock_data_wrapper(ticker: str) -> Dict[str, Any]:
            try:
                return stock_data_tool({"ticker": ticker})
            except Exception as e:
                return {"error": f"Failed to fetch stock data: {str(e)}"}
            
        def financial_metrics_wrapper(ticker: str) -> Dict[str, Any]:
            try:
                return financial_metrics_tool({"ticker": ticker})
            except Exception as e:
                return {"error": f"Failed to fetch financial metrics: {str(e)}"}

        # Configure tools with proper schemas
        self.tools = {
            "search": Tool(
                name="Search",
                func=search_wrapper,
                description="Search the internet for recent information. Input should be a simple search query string."
            ),
            "stock_data": Tool(
                name="StockData",
                func=stock_data_wrapper,
                description="Get current stock data and basic financials. Input should be a stock ticker symbol."
            ),
            "financial_metrics": Tool(
                name="FinancialMetrics",
                func=financial_metrics_wrapper,
                description="Get detailed financial metrics and ratios. Input should be a stock ticker symbol."
            )
        }

    def create_agents(self):
        # Update agent configurations with proper tool assignments
        market_researcher = Agent(
            role='Market Intelligence Officer',
            goal='Provide actionable market research and competitive analysis',
            backstory="""Expert market researcher focused on industry trends and competitive analysis. 
            Specializes in identifying market opportunities and risks.""",
            tools=[self.tools["search"], self.tools["stock_data"]],
            llm=llm,
            verbose=True
        )

        technical_analyst = Agent(
            role='Technical Analysis Specialist',
            goal='Provide technical analysis and price targets',
            backstory="""Technical analysis expert specializing in price patterns and momentum indicators. 
            Focuses on identifying entry/exit points based on technical signals.""",
            tools=[self.tools["stock_data"]],
            llm=llm,
            verbose=True
        )

        fundamental_analyst = Agent(
            role='Fundamental Analysis Expert',
            goal='Evaluate financial health and provide valuation analysis',
            backstory="""Financial analysis expert specializing in company valuations and financial metrics. 
            Focuses on analyzing financial statements and ratios.""",
            tools=[self.tools["stock_data"], self.tools["financial_metrics"]],
            llm=llm,
            verbose=True
        )

        strategy_expert = Agent(
            role='Portfolio Strategy Expert',
            goal='Synthesize all analyses into actionable investment recommendations',
            backstory="""Investment strategist who combines market, technical, and fundamental analysis 
            to create comprehensive investment strategies.""",
            tools=[self.tools["stock_data"]],
            llm=llm,
            verbose=True
        )

        return [market_researcher, technical_analyst, fundamental_analyst, strategy_expert]

    def create_tasks(self, ticker, agents):
        [market_researcher, technical_analyst, fundamental_analyst, strategy_expert] = agents

        market_research_task = Task(
            description=f"""Research {ticker} market conditions and competitive landscape.
            Focus on:
            1. Current market position and trends
            2. Competitive advantages and threats
            3. Industry dynamics and market share
            4. Recent news and developments
            5. Market sentiment analysis
            
            Use the search tool with simple text queries and the stock data tool with the ticker symbol.
            
            Provide a detailed JSON response with your findings.""",
            expected_output="""A JSON object containing:
            {
                "market_position": "Current market standing and trends",
                "competitive_analysis": {
                    "advantages": ["List of competitive advantages"],
                    "threats": ["List of potential threats"]
                },
                "industry_analysis": "Industry dynamics and market share details",
                "recent_developments": ["List of recent news and events"],
                "market_sentiment": "Overall market sentiment analysis"
            }""",
            agent=market_researcher
        )

        technical_analysis_task = Task(
            description=f"""Analyze {ticker} technical indicators and price patterns.
            Focus on:
            1. Current price trends and momentum
            2. Support and resistance levels
            3. Volume analysis
            4. Technical indicators (RSI, MACD)
            5. Price targets
            
            Use the stock data tool with the ticker symbol.
            
            Provide specific entry/exit points in your JSON response.""",
            expected_output="""A JSON object containing:
            {
                "trend_analysis": "Current price trend analysis",
                "support_resistance": {
                    "support_levels": ["List of support prices"],
                    "resistance_levels": ["List of resistance prices"]
                },
                "volume_analysis": "Trading volume analysis",
                "technical_indicators": {
                    "rsi": "RSI value and interpretation",
                    "macd": "MACD analysis"
                },
                "price_targets": {
                    "entry_points": ["List of recommended entry prices"],
                    "exit_points": ["List of recommended exit prices"],
                    "target_price": "Price target"
                }
            }""",
            agent=technical_analyst
        )

        fundamental_analysis_task = Task(
            description=f"""Evaluate {ticker} fundamental metrics and valuation.
            Focus on:
            1. Financial health indicators
            2. Growth metrics and trends
            3. Profitability ratios
            4. Valuation metrics
            5. Risk assessment
            
            Use both stock data and financial metrics tools with the ticker symbol.
            
            Provide a detailed valuation analysis in your JSON response.""",
            expected_output="""A JSON object containing:
            {
                "financial_health": {
                    "liquidity_ratios": "Analysis of liquidity",
                    "solvency_ratios": "Analysis of solvency",
                    "overall_health": "Overall financial health assessment"
                },
                "growth_metrics": {
                    "revenue_growth": "Revenue growth analysis",
                    "earnings_growth": "Earnings growth analysis",
                    "future_outlook": "Growth outlook"
                },
                "profitability": {
                    "margins": "Margin analysis",
                    "returns": "Return metrics analysis"
                },
                "valuation": {
                    "current_valuation": "Current valuation metrics",
                    "fair_value": "Calculated fair value",
                    "valuation_assessment": "Over/undervalued assessment"
                },
                "risk_analysis": "Comprehensive risk assessment"
            }""",
            agent=fundamental_analyst
        )

        strategy_task = Task(
            description=f"""Create an investment strategy for {ticker} based on all analyses.
            Focus on:
            1. Investment recommendation
            2. Position sizing
            3. Risk management
            4. Entry/exit strategy
            5. Portfolio considerations
            
            Use the stock data tool with the ticker symbol.
            
            Provide a comprehensive strategy in your JSON response.""",
            expected_output="""A JSON object containing:
            {
                "recommendation": {
                    "action": "Buy/Sell/Hold recommendation",
                    "confidence": "Confidence level in recommendation",
                    "time_horizon": "Recommended investment timeframe"
                },
                "position_sizing": {
                    "recommended_size": "Recommended position size",
                    "rationale": "Rationale for position size"
                },
                "risk_management": {
                    "stop_loss": "Stop loss price and rationale",
                    "risk_reward_ratio": "Risk/reward ratio",
                    "max_drawdown": "Maximum acceptable drawdown"
                },
                "execution_strategy": {
                    "entry_strategy": "Detailed entry strategy",
                    "exit_strategy": "Detailed exit strategy",
                    "monitoring_points": ["Key points to monitor"]
                },
                "portfolio_fit": "Analysis of how this fits into a portfolio"
            }""",
            agent=strategy_expert
        )

        return [market_research_task, technical_analysis_task, fundamental_analysis_task, strategy_task]

    def analyze_stock(self, ticker):
        """
        Perform comprehensive stock analysis using the specialized crew
        """
        try:
            agents = self.create_agents()
            tasks = self.create_tasks(ticker, agents)

            crew = Crew(
                agents=agents,
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )

            # Get crew output
            crew_output = crew.kickoff()

            # Process and structure the results using documented CrewOutput attributes
            processed_result = {}

            # Get raw output
            processed_result["raw_output"] = crew_output.raw

            # Get JSON output if available
            if crew_output.json_dict:
                processed_result["json_output"] = crew_output.json_dict

            # Get Pydantic output if available
            if crew_output.pydantic:
                processed_result["pydantic_output"] = crew_output.pydantic

            # Get tasks output
            processed_result["tasks_output"] = crew_output.tasks_output

            # Get token usage
            processed_result["token_usage"] = crew_output.token_usage

            return processed_result

        except Exception as e:
            print(f"Analysis failed with error: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "raw_output": None,
                "json_output": None,
                "tasks_output": None,
                "token_usage": None
            } 