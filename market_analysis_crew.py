from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from financial_tools import stock_data_tool, financial_metrics_tool
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
import queue
import threading
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from litellm import APIConnectionError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add retry decorator for LLM initialization
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(APIConnectionError),
    reraise=True
)
def initialize_llm():
    """Initialize LLM with retry mechanism"""
    return LLM(
        model="gemini/gemini-1.5-pro-latest",
        temperature=0.7
    )

# Update the LLM initialization
logger.info("Initializing LLM...")
llm = initialize_llm()

# Global message queue for agent updates
agent_messages = queue.Queue()

class AgentCallback:
    """Callback handler for agent progress updates"""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def on_start(self, task: str):
        self._add_message(f"Starting task: {task}", "start")

    def on_progress(self, message: str):
        self._add_message(message, "progress")

    def on_complete(self, result: Any):
        self._add_message("Task completed", "complete", result)

    def _add_message(self, message: str, status: str, result: Any = None):
        timestamp = datetime.now().isoformat()
        agent_messages.put({
            "timestamp": timestamp,
            "agent": self.agent_name,
            "message": message,
            "status": status,
            "result": result
        })

class EnhancedAgent(Agent):
    """Enhanced Agent class with progress tracking and retry mechanism"""
    def __init__(self, role: str, goal: str, backstory: str, tools: List[Tool], llm: Any, verbose: bool = True):
        super().__init__(role=role, goal=goal, backstory=backstory, tools=tools, llm=llm, verbose=verbose)
        self._callback = AgentCallback(role)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(APIConnectionError),
        reraise=True
    )
    def execute_task(self, task: Task, context: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Execute a task with progress tracking and retry mechanism
        
        Args:
            task: The task to execute
            context: Optional context dictionary for the task
            **kwargs: Additional keyword arguments
            
        Returns:
            str: The result of the task execution
        """
        self._callback.on_start(task.description)
        try:
            # Simulate thinking/processing time (remove in production)
            time.sleep(2)
            self._callback.on_progress("Analyzing data...")
            time.sleep(1)
            
            # Call parent's execute_task with proper arguments
            result = super().execute_task(task=task, context=context)
            self._callback.on_complete(result)
            return result
        except Exception as e:
            self._callback.on_progress(f"Error: {str(e)}")
            raise

class RiskManager:
    @staticmethod
    def calculate_volatility(prices: List[float]) -> float:
        """Calculate historical volatility"""
        returns = pd.Series(prices).pct_change().dropna()
        return returns.std() * np.sqrt(252)  # Annualized volatility
    
    @staticmethod
    def calculate_var(prices: List[float], confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        returns = pd.Series(prices).pct_change().dropna()
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_sharpe_ratio(prices: List[float], risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe Ratio"""
        returns = pd.Series(prices).pct_change().dropna()
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def assess_market_risk(market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive market risk assessment"""
        prices = market_data.get("price_history", [])
        if not prices:
            return {"error": "Insufficient price data"}
        
        return {
            "volatility": RiskManager.calculate_volatility(prices),
            "value_at_risk": RiskManager.calculate_var(prices),
            "sharpe_ratio": RiskManager.calculate_sharpe_ratio(prices),
            "beta": market_data.get("beta"),
            "risk_assessment": "high" if RiskManager.calculate_volatility(prices) > 0.3 else "medium" if RiskManager.calculate_volatility(prices) > 0.15 else "low"
        }

class MarketAnalysisCrew:
    def __init__(self):
        logger.info("Initializing Market Analysis Crew")
        self._init_tools()
        self.risk_manager = RiskManager()
        self.messages = []

    def _init_tools(self):
        # Initialize tools with proper error handling
        def search_wrapper(query: str) -> str:
            try:
                search = DuckDuckGoSearchRun()
                return search.run(query)
            except Exception as e:
                logger.error(f"Search failed: {str(e)}")
                return f"Search failed: {str(e)}"
        
        def stock_data_wrapper(ticker: str) -> Dict[str, Any]:
            try:
                return stock_data_tool({"ticker": ticker})
            except Exception as e:
                logger.error(f"Failed to fetch stock data: {str(e)}")
                return {"error": f"Failed to fetch stock data: {str(e)}"}
            
        def financial_metrics_wrapper(ticker: str) -> Dict[str, Any]:
            try:
                return financial_metrics_tool({"ticker": ticker})
            except Exception as e:
                logger.error(f"Failed to fetch financial metrics: {str(e)}")
                return {"error": f"Failed to fetch financial metrics: {str(e)}"}

        self.tools = {
            "search": Tool(
                name="Search",
                func=search_wrapper,
                description="Search the internet for recent information. Input should be a simple search query string."
            ),
            "stock_data": Tool(
                name="StockData",
                func=stock_data_wrapper,
                description="Get comprehensive stock data including technical indicators. Input should be a stock ticker symbol."
            ),
            "financial_metrics": Tool(
                name="FinancialMetrics",
                func=financial_metrics_wrapper,
                description="Get detailed financial metrics and analysis. Input should be a stock ticker symbol."
            )
        }

    def create_agents(self) -> List[EnhancedAgent]:
        logger.info("Creating specialized agents")
        
        market_researcher = EnhancedAgent(
            role='Market Intelligence Officer',
            goal='Provide actionable market research and competitive analysis',
            backstory="""Expert market researcher with over 15 years of experience in industry analysis.
            Specializes in identifying market opportunities, competitive advantages, and potential risks.
            Known for combining quantitative and qualitative analysis to form comprehensive market views.""",
            tools=[self.tools["search"], self.tools["stock_data"]],
            llm=llm,
            verbose=True
        )

        technical_analyst = EnhancedAgent(
            role='Technical Analysis Specialist',
            goal='Provide sophisticated technical analysis and precise price targets',
            backstory="""Certified Technical Analyst with expertise in advanced pattern recognition and momentum analysis.
            Masters multiple timeframe analysis and combines various technical indicators for high-probability setups.
            Specializes in risk management and position sizing based on technical levels.""",
            tools=[self.tools["stock_data"]],
            llm=llm,
            verbose=True
        )

        fundamental_analyst = EnhancedAgent(
            role='Fundamental Analysis Expert',
            goal='Conduct deep fundamental analysis and accurate valuation assessment',
            backstory="""CFA charterholder with extensive experience in equity research and valuation.
            Expert in financial statement analysis, industry comparison, and intrinsic value calculation.
            Specializes in identifying companies with strong competitive advantages and growth potential.""",
            tools=[self.tools["stock_data"], self.tools["financial_metrics"]],
            llm=llm,
            verbose=True
        )

        risk_analyst = EnhancedAgent(
            role='Risk Management Specialist',
            goal='Assess and quantify investment risks',
            backstory="""Risk management expert with background in quantitative finance.
            Specializes in portfolio risk assessment, volatility analysis, and risk-adjusted returns.
            Expert in using various risk metrics and stress testing scenarios.""",
            tools=[self.tools["stock_data"], self.tools["financial_metrics"]],
            llm=llm,
            verbose=True
        )

        strategy_expert = EnhancedAgent(
            role='Portfolio Strategy Expert',
            goal='Synthesize all analyses into actionable investment recommendations',
            backstory="""Senior Portfolio Manager with 20+ years of investment experience.
            Expert in developing comprehensive investment strategies that balance risk and reward.
            Specializes in portfolio optimization and risk-adjusted position sizing.""",
            tools=[self.tools["stock_data"]],
            llm=llm,
            verbose=True
        )

        return [market_researcher, technical_analyst, fundamental_analyst, risk_analyst, strategy_expert]

    def get_agent_messages(self) -> List[Dict[str, Any]]:
        """Get all agent messages from the queue"""
        messages = []
        while not agent_messages.empty():
            messages.append(agent_messages.get())
        return messages

    def create_tasks(self, ticker: str, agents: List[EnhancedAgent]) -> List[Task]:
        logger.info(f"Creating analysis tasks for ticker: {ticker}")
        [market_researcher, technical_analyst, fundamental_analyst, risk_analyst, strategy_expert] = agents

        market_research_task = Task(
            description=f"""Conduct comprehensive market research for {ticker}.
            Focus on:
            1. Industry position and market share analysis
            2. Competitive advantage assessment
            3. Market trends and growth opportunities
            4. Regulatory environment and potential impacts
            5. ESG considerations and sustainability
            
            Provide detailed analysis with specific metrics and data points.""",
            expected_output="""A detailed market research report in JSON format containing:
            {
                "industry_analysis": {
                    "market_position": "Analysis of company's market position",
                    "market_share": "Estimated market share percentage",
                    "competitors": ["List of main competitors"]
                },
                "competitive_advantages": {
                    "strengths": ["List of key competitive advantages"],
                    "moat": "Description of economic moat",
                    "sustainability": "Analysis of advantage sustainability"
                },
                "market_trends": {
                    "industry_growth": "Industry growth rate and trends",
                    "opportunities": ["List of growth opportunities"],
                    "threats": ["List of potential threats"]
                },
                "regulatory_analysis": {
                    "current_regulations": "Key regulatory requirements",
                    "upcoming_changes": "Potential regulatory changes",
                    "compliance_status": "Company's compliance standing"
                },
                "esg_profile": {
                    "environmental": "Environmental impact and initiatives",
                    "social": "Social responsibility measures",
                    "governance": "Corporate governance assessment"
                }
            }""",
            agent=market_researcher
        )

        technical_analysis_task = Task(
            description=f"""Perform advanced technical analysis for {ticker}.
            Focus on:
            1. Multi-timeframe trend analysis
            2. Key support and resistance levels
            3. Volume profile and momentum indicators
            4. Pattern recognition and probable scenarios
            5. Risk levels and position sizing recommendations
            
            Include specific price levels and probability assessments.""",
            expected_output="""A comprehensive technical analysis in JSON format containing:
            {
                "trend_analysis": {
                    "primary_trend": "Current primary trend direction",
                    "secondary_trend": "Current secondary trend direction",
                    "trend_strength": "Trend strength assessment"
                },
                "support_resistance": {
                    "support_levels": ["List of key support prices"],
                    "resistance_levels": ["List of key resistance prices"],
                    "breakout_points": ["Critical price levels to watch"]
                },
                "technical_indicators": {
                    "moving_averages": "MA analysis and crossovers",
                    "momentum_indicators": {
                        "rsi": "RSI analysis and signals",
                        "macd": "MACD analysis and signals"
                    },
                    "volume_analysis": "Volume trend and significance"
                },
                "patterns": {
                    "identified_patterns": ["List of recognized patterns"],
                    "probability": "Success probability assessment",
                    "target_prices": ["Projected price targets"]
                },
                "position_recommendations": {
                    "entry_points": ["Recommended entry levels"],
                    "stop_loss": ["Suggested stop-loss levels"],
                    "position_size": "Recommended position sizing"
                }
            }""",
            agent=technical_analyst
        )

        fundamental_analysis_task = Task(
            description=f"""Conduct detailed fundamental analysis for {ticker}.
            Focus on:
            1. Financial statement analysis and quality of earnings
            2. Valuation using multiple methods (DCF, Multiples, etc.)
            3. Capital structure and efficiency metrics
            4. Growth sustainability assessment
            5. Competitive advantage period estimation
            
            Provide specific metrics and comparative analysis.""",
            expected_output="""A detailed fundamental analysis in JSON format containing:
            {
                "financial_analysis": {
                    "income_statement": "Key income metrics and trends",
                    "balance_sheet": "Balance sheet strength assessment",
                    "cash_flow": "Cash flow analysis and quality",
                    "earnings_quality": "Earnings quality assessment"
                },
                "valuation": {
                    "dcf_valuation": {
                        "fair_value": "Calculated fair value per share",
                        "assumptions": "Key assumptions used",
                        "sensitivity": "Sensitivity analysis"
                    },
                    "relative_valuation": {
                        "peer_comparison": "Industry peer analysis",
                        "multiple_analysis": "Key valuation multiples"
                    }
                },
                "capital_structure": {
                    "debt_analysis": "Debt level and coverage",
                    "equity_analysis": "Equity structure assessment",
                    "efficiency_metrics": "Capital efficiency ratios"
                },
                "growth_analysis": {
                    "historical_growth": "Past growth rates",
                    "future_prospects": "Growth sustainability assessment",
                    "reinvestment_needs": "Capital requirements"
                },
                "competitive_position": {
                    "advantage_period": "Estimated competitive advantage duration",
                    "moat_analysis": "Economic moat assessment",
                    "industry_position": "Competitive position strength"
                }
            }""",
            agent=fundamental_analyst
        )

        risk_analysis_task = Task(
            description=f"""Perform comprehensive risk assessment for {ticker}.
            Focus on:
            1. Volatility and Value at Risk analysis
            2. Market risk factors and sensitivity
            3. Company-specific risk factors
            4. Industry and systematic risk exposure
            5. Scenario analysis and stress testing
            
            Quantify risks where possible and provide mitigation strategies.""",
            expected_output="""A comprehensive risk assessment in JSON format containing:
            {
                "market_risk": {
                    "volatility_analysis": "Historical and implied volatility",
                    "var_analysis": "Value at Risk calculations",
                    "beta": "Market sensitivity assessment",
                    "correlation": "Market correlation analysis"
                },
                "company_risks": {
                    "operational_risks": ["Key operational risk factors"],
                    "financial_risks": ["Financial risk exposures"],
                    "management_risks": ["Management and governance risks"]
                },
                "industry_risks": {
                    "sector_risks": ["Industry-specific risk factors"],
                    "competitive_risks": ["Competition-related risks"],
                    "regulatory_risks": ["Regulatory risk exposure"]
                },
                "scenario_analysis": {
                    "stress_tests": ["Stress test scenarios and impacts"],
                    "sensitivity": ["Key factor sensitivities"],
                    "worst_case": "Worst-case scenario analysis"
                },
                "risk_mitigation": {
                    "strategies": ["Risk mitigation recommendations"],
                    "hedging": "Hedging opportunities",
                    "monitoring": ["Key risk indicators to monitor"]
                }
            }""",
            agent=risk_analyst
        )

        strategy_task = Task(
            description=f"""Develop comprehensive investment strategy for {ticker}.
            Focus on:
            1. Investment thesis and conviction level
            2. Position sizing and risk allocation
            3. Entry and exit strategy with specific levels
            4. Risk management parameters
            5. Portfolio context and correlation analysis
            
            Provide actionable recommendations with specific parameters.""",
            expected_output="""A detailed investment strategy in JSON format containing:
            {
                "investment_thesis": {
                    "summary": "Core investment thesis",
                    "conviction": "Conviction level assessment",
                    "time_horizon": "Recommended investment timeframe",
                    "expected_return": "Target return projection"
                },
                "position_strategy": {
                    "size_recommendation": "Recommended position size",
                    "allocation_logic": "Position sizing rationale",
                    "portfolio_fit": "Portfolio context analysis"
                },
                "execution_plan": {
                    "entry_strategy": {
                        "price_levels": ["Specific entry price points"],
                        "timing": "Entry timing considerations",
                        "method": "Recommended entry method"
                    },
                    "exit_strategy": {
                        "profit_targets": ["Specific profit-taking levels"],
                        "stop_losses": ["Stop-loss price points"],
                        "rebalancing": "Position rebalancing rules"
                    }
                },
                "risk_management": {
                    "position_limits": "Maximum position size",
                    "loss_limits": "Maximum acceptable loss",
                    "monitoring_plan": ["Key metrics to monitor"]
                },
                "portfolio_impact": {
                    "correlation": "Portfolio correlation analysis",
                    "diversification": "Diversification impact",
                    "risk_contribution": "Risk contribution assessment"
                }
            }""",
            agent=strategy_expert
        )

        return [market_research_task, technical_analysis_task, fundamental_analysis_task, risk_analysis_task, strategy_task]

    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive stock analysis using the specialized crew
        """
        logger.info(f"Starting comprehensive analysis for ticker: {ticker}")
        try:
            # Clear previous messages
            while not agent_messages.empty():
                agent_messages.get()

            # Create agents and tasks
            agents = self.create_agents()
            tasks = self.create_tasks(ticker, agents)
            
            # Create crew with sequential process
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Execute analysis
            result = crew.kickoff()
            
            # Parse and structure results
            analysis_results = self._parse_results(result)
            
            # Add risk metrics
            stock_data = self.tools["stock_data"].func(ticker)
            risk_metrics = self.risk_manager.assess_market_risk(stock_data)
            analysis_results["risk_metrics"] = risk_metrics
            
            # Log success
            logger.info(f"Successfully completed analysis for {ticker}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error during stock analysis: {str(e)}")
            raise

    def _parse_results(self, result: str) -> Dict[str, Any]:
        """Parse and structure the analysis results"""
        try:
            # Attempt to parse JSON if result is in JSON format
            return json.loads(result)
        except:
            # If not JSON, return as structured text
            return {
                "raw_analysis": result,
                "timestamp": datetime.now().isoformat()
            } 