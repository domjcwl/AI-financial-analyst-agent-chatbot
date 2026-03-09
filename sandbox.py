from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os
import requests


load_dotenv()
search_tool = DuckDuckGoSearchRun()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

@tool
def smart_news_search(query: str, ticker: str = None) -> str: 
    """ 
    Finds news on the web. - Use 'ticker' if searching for a specific company (e.g., 'AAPL', 'TSLA'). - Use 'query' for general topics (e.g., 'inflation', 'tech trends'). 
    """ 
    print("="*50, f"Fetching financial news for {query}:", "="*50) 
    return search_tool.run(f"{ticker} {query}" if (ticker and query) else query)


@tool
def get_financial_statements(ticker: str, period: str = "quarterly", limit: int = 1):
    """
    Fetch income statement, balance sheet, and cash flow statement.
    """
    print("="*50, "Fetching financial statements", "="*50)
    print("="*50,f"Ticker: {ticker}, Period: {period}, Limit: {limit}", "="*50)
    endpoints = {
        "income_statement": f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&period={period}&limit={limit}&apikey={FMP_API_KEY}",
        "balance_sheet": f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={ticker}&period={period}&limit={limit}&apikey={FMP_API_KEY}",
        "cash_flow": f"https://financialmodelingprep.com/stable/cash-flow-statement?symbol={ticker}&period={period}&limit={limit}&apikey={FMP_API_KEY}",
    }

    results = {}

    for name, url in endpoints.items():
        response = requests.get(url, timeout=5 )
        if response.status_code == 200:
            results[name] = response.json()
        else:
            results[name] = f"Error: {response.status_code}"

    return results



tools = [get_financial_statements, smart_news_search]

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=OPENAI_API_KEY)

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

result = agent.invoke({
    "messages": [{"role": "user", "content": """ tell me more about the JP morgan stock
                                                 """}]
})

print(result["messages"][-1].content)  # Final answer



