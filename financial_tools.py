from langchain.tools import Tool
import yfinance as yf
from typing import Dict, Any, Optional, List
import json
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
from functools import wraps
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def retry_on_exception(retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:
                        logger.error(f"Failed after {retries} retries: {str(e)}")
                        raise
                    logger.warning(f"Attempt {i+1} failed: {str(e)}, retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class FinancialMetricsCalculator:
    @staticmethod
    def calculate_moving_averages(prices: List[float], windows: List[int]) -> Dict[str, List[float]]:
        df = pd.Series(prices)
        return {f"MA{window}": df.rolling(window=window).mean().tolist() for window in windows}
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        df = pd.Series(prices)
        delta = df.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.tolist()
    
    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict[str, List[float]]:
        df = pd.Series(prices)
        exp1 = df.ewm(span=12, adjust=False).mean()
        exp2 = df.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return {
            "macd": macd.tolist(),
            "signal": signal.tolist(),
            "histogram": (macd - signal).tolist()
        }

@retry_on_exception()
def stock_data_tool(args: Dict[str, str]) -> Dict[str, Any]:
    """
    Enhanced stock data tool with technical indicators and error handling
    """
    logger.info(f"Fetching stock data for ticker: {args.get('ticker')}")
    try:
        ticker = args.get("ticker")
        if not ticker:
            raise ValueError("No ticker provided")
        
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1y")
        
        prices = history["Close"].tolist()
        calculator = FinancialMetricsCalculator()
        
        technical_indicators = {
            "moving_averages": calculator.calculate_moving_averages(prices, [20, 50, 200]),
            "rsi": calculator.calculate_rsi(prices),
            "macd": calculator.calculate_macd(prices)
        }
        
        result = {
            "current_price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "volume": info.get("volume"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "price_history": prices,
            "volume_history": history["Volume"].tolist(),
            "dates": history.index.strftime('%Y-%m-%d').tolist(),
            "technical_indicators": technical_indicators
        }
        
        logger.info(f"Successfully fetched stock data for {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        raise

@retry_on_exception()
def financial_metrics_tool(args: Dict[str, str]) -> Dict[str, Any]:
    """
    Enhanced financial metrics tool with comprehensive analysis
    """
    logger.info(f"Calculating financial metrics for ticker: {args.get('ticker')}")
    try:
        ticker = args.get("ticker")
        if not ticker:
            raise ValueError("No ticker provided")
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get financial statements
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow
        
        result = {
            "profitability": {
                "gross_margin": info.get("grossMargins"),
                "operating_margin": info.get("operatingMargins"),
                "profit_margin": info.get("profitMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "roic": info.get("returnOnCapital")
            },
            "valuation": {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "peg_ratio": info.get("pegRatio")
            },
            "growth": {
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
                "free_cashflow_growth": calculate_growth_rate(cash_flow, "Free Cash Flow")
            },
            "financial_health": {
                "current_ratio": info.get("currentRatio"),
                "debt_to_equity": info.get("debtToEquity"),
                "quick_ratio": info.get("quickRatio"),
                "total_debt": info.get("totalDebt"),
                "total_cash": info.get("totalCash"),
                "interest_coverage": calculate_interest_coverage(income_stmt)
            },
            "efficiency": {
                "asset_turnover": calculate_asset_turnover(income_stmt, balance_sheet),
                "inventory_turnover": info.get("inventoryTurnover"),
                "receivables_turnover": calculate_receivables_turnover(income_stmt, balance_sheet)
            }
        }
        
        logger.info(f"Successfully calculated financial metrics for {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating financial metrics: {str(e)}")
        raise

def calculate_growth_rate(df: pd.DataFrame, column: str) -> float:
    """Calculate year-over-year growth rate"""
    if df.empty or column not in df.columns:
        return None
    latest = df[column].iloc[0]
    previous = df[column].iloc[4] if len(df) > 4 else df[column].iloc[-1]
    return ((latest - previous) / abs(previous)) if previous != 0 else None

def calculate_interest_coverage(income_stmt: pd.DataFrame) -> float:
    """Calculate interest coverage ratio"""
    if income_stmt.empty:
        return None
    try:
        ebit = income_stmt.loc["Operating Income"].iloc[0]
        interest_expense = abs(income_stmt.loc["Interest Expense"].iloc[0])
        return ebit / interest_expense if interest_expense != 0 else None
    except:
        return None

def calculate_asset_turnover(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> float:
    """Calculate asset turnover ratio"""
    if income_stmt.empty or balance_sheet.empty:
        return None
    try:
        revenue = income_stmt.loc["Total Revenue"].iloc[0]
        avg_assets = (balance_sheet.loc["Total Assets"].iloc[0] + balance_sheet.loc["Total Assets"].iloc[1]) / 2
        return revenue / avg_assets if avg_assets != 0 else None
    except:
        return None

def calculate_receivables_turnover(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> float:
    """Calculate receivables turnover ratio"""
    if income_stmt.empty or balance_sheet.empty:
        return None
    try:
        revenue = income_stmt.loc["Total Revenue"].iloc[0]
        avg_receivables = (balance_sheet.loc["Net Receivables"].iloc[0] + balance_sheet.loc["Net Receivables"].iloc[1]) / 2
        return revenue / avg_receivables if avg_receivables != 0 else None
    except:
        return None

stock_data_tool_instance = Tool(
    name="stock_data_tool",
    func=stock_data_tool,
    description="Fetches comprehensive stock data including technical indicators"
)

financial_metrics_tool_instance = Tool(
    name="financial_metrics_tool",
    func=financial_metrics_tool,
    description="Calculates detailed financial metrics with comprehensive analysis"
) 