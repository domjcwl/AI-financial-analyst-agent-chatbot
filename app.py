import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
import httpx

custom_client = httpx.Client(verify=False)

# 1. Setup & Keys
load_dotenv()
search_tool = DuckDuckGoSearchRun()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINANCIAL_DATASETS_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")

# 2. Tool Definitions
@tool
def smart_news_search(query: str, ticker: str = None) -> str: 
    """ 
    Finds news on the web. 
    - Use 'ticker' if searching for a specific company (e.g., 'AAPL', 'TSLA'). 
    - Use 'query' for general topics (e.g., 'inflation', 'tech trends'). 
    
    
    use smart_news_search to analyse the market sentiment, industry trends, general news, and more to aid in response if needed.
    """ 
    print(f"\n[System] Searching news for: {query}...") 
    return search_tool.run(f"{ticker} {query}" if (ticker and query) else query)

@tool
def get_financial_statements(ticker: str, period: str = "quarterly", limit: int = 1):
    """
    Fetch income statement, balance sheet, and cash flow statement for a specific ticker.
    """
    print(f"\n[System] Fetching financials for {ticker}...")
    endpoints = {
        "income_statement": f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&period={period}&limit={limit}&apikey={FMP_API_KEY}",
        "balance_sheet": f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={ticker}&period={period}&limit={limit}&apikey={FMP_API_KEY}",
        "cash_flow": f"https://financialmodelingprep.com/stable/cash-flow-statement?symbol={ticker}&period={period}&limit={limit}&apikey={FMP_API_KEY}",
    }

    results = {}
    for name, url in endpoints.items():
        response = requests.get(url, timeout=5)
        results[name] = response.json() if response.status_code == 200 else f"Error: {response.status_code}"
    return results

@tool
def get_analyst_data(ticker: str) -> dict:
    """
    Fetch analyst sentiment data for a stock ticker.

    Use when the user asks about:
    - analysis of stock
        """

    print(f"\n[System] Fetching analyst data for {ticker}...")

    base = "https://finnhub.io/api/v1"
    token = FINNHUB_API_KEY

    endpoints = {
        "recommendations": f"{base}/stock/recommendation?symbol={ticker}&token={token}",
        "price_target": f"{base}/stock/price-target?symbol={ticker}&token={token}",
        "rating_changes": f"{base}/stock/upgrade-downgrade?symbol={ticker}&token={token}",
    }

    results = {}

    try:
        for name, url in endpoints.items():
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                results[name] = response.json()
            else:
                results[name] = f"Error: {response.status_code}"

        # Optional: only return latest recommendation
        if isinstance(results.get("recommendations"), list) and results["recommendations"]:
            results["latest_recommendation"] = results["recommendations"][0]

        # Optional: limit rating changes to recent ones
        if isinstance(results.get("rating_changes"), list):
            results["rating_changes"] = results["rating_changes"][:10]

        return results

    except Exception as e:
        return {"error": str(e)}
    
@tool 
def get_key_metrics():#refer to gemini pinned chat to do
    '''random docstring'''
    print("hi world")

# 3. Agent Initialization
tools = [get_financial_statements, smart_news_search, get_analyst_data,]
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=OPENAI_API_KEY, http_client=custom_client)

system_prompt = """You are a helpful AI financial analyst. Your goal is to provide deep, data-driven 
insights. Always ask the user for a follow-up prompt relevant to your answer.

Rules:
1. For 'get_financial_statements': Provide deep analysis on metrics, trends, and stock price impact.
2. For 'smart_news_search': Include SOURCE and DATE. Provide a variety of news (earnings, trends, events)."""

# Using create_react_agent to handle the conversation state properly
agent = create_agent(
    llm, 
    tools, 
    system_prompt = """You are a helpful AI financial analyst. Your goal is to provide deep, data-driven 
insights, and elaborate on topics given. Always ask the user for a follow up prompt that is relevant to the answer that you gave or relevant to the user's question/prompt.

Follow these tool-specific rules:

1. For 'get_financial_statements':
    - Always fetch the latest financial statements for the specified company.
    - Always include the period (quarterly or annual) and the number of periods requested in the final answer.
    - Always go indepth on the financial data, providing a comprehensive analysis of the key metrics, trends, and implications for the stock or market.
    - Always provide insights on how the financial performance may impact the stock price and investor sentiment.
    - Always provide a clear and concise summary of the financial health of the company based on the retrieved statements.
    
2. For 'smart_news_search':
   - Always dig deep into the news and explain each key portion of the news article with great depth, providing a comprehensive summary and analysis of the key points, implications, and potential impact on the stock or market.
   - Always include the SOURCE OF THE NEWS and the DATE OF THE SOURCE OF THE NEWS for each news article in the final answer.
   - Always provide the latest news, ensuring that the information is up-to-date and relevant to the current market conditions.
   - Always provide a good and vast variety of news, covering different aspects such as earnings reports, market trends, analyst opinions, and any significant events that could affect the stock or market.
"""
)

# 4. Continuous Loop
def start_chat():
    # This list will store the entire conversation history
    messages = []
    
    print("--- Financial Analyst AI Active ---")
    print("(Type 'exit' or 'quit' to stop)\n")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chat ended. Goodbye!")
            break

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Invoke agent with the full message history
        response = agent.invoke({"messages": messages})
        
        # Update history with the full list of messages (including AI response and tool outputs)
        messages = response["messages"]

        # Print the last message (the AI's final answer)
        print(f"\nAI: {messages[-1].content}\n")
        print("-" * 30)

if __name__ == "__main__":
    start_chat()