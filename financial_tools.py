from langchain.tools import Tool
import yfinance as yf
from typing import Dict, Any
import json

def stock_data_tool(args: Dict[str, str]) -> Dict[str, Any]:
    """
    Get current stock data and basic financials
    """
    try:
        ticker = args.get("ticker")
        if not ticker:
            return {"error": "No ticker provided"}
        
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1y")
        
        return {
            "current_price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "volume": info.get("volume"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "price_history": history["Close"].tolist(),
            "volume_history": history["Volume"].tolist(),
            "dates": history.index.strftime('%Y-%m-%d').tolist()
        }
    except Exception as e:
        return {"error": f"Failed to fetch stock data: {str(e)}"}

def financial_metrics_tool(args: Dict[str, str]) -> Dict[str, Any]:
    """
    Get detailed financial metrics and ratios
    """
    try:
        ticker = args.get("ticker")
        if not ticker:
            return {"error": "No ticker provided"}
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "profitability": {
                "gross_margin": info.get("grossMargins"),
                "operating_margin": info.get("operatingMargins"),
                "profit_margin": info.get("profitMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets")
            },
            "valuation": {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda")
            },
            "growth": {
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth")
            },
            "financial_health": {
                "current_ratio": info.get("currentRatio"),
                "debt_to_equity": info.get("debtToEquity"),
                "quick_ratio": info.get("quickRatio"),
                "total_debt": info.get("totalDebt"),
                "total_cash": info.get("totalCash")
            }
        }
    except Exception as e:
        return {"error": f"Failed to fetch financial metrics: {str(e)}"}

# Create tool instances
stock_data_tool_instance = Tool(
    name="stock_data_tool",
    func=stock_data_tool,
    description="Fetches current stock data and basic financials"
)

financial_metrics_tool_instance = Tool(
    name="financial_metrics_tool",
    func=financial_metrics_tool,
    description="Calculates detailed financial metrics and ratios"
) 