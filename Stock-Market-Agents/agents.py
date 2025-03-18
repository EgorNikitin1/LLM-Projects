from phi.agent import Agent, RunResponse
from phi.model.ollama import Ollama
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools
from phi.utils.pprint import pprint_run_response

from datetime import datetime, timedelta

## Create Agents

# Sentiment Agent
sentiment_agent = Agent(
    name="Sentiment Agent",
    role="Search and interpret news articles.",
    model=Ollama(id="llama3.1"),
    tools=[GoogleSearch()],
    instructions=[
        "Find relevant news articles for each company and analyze the sentiment.",
        "Provide sentiment scores from 1 (negative) to 10 (positive) with reasoning and sources."
        "Cite your sources. Be specific and provide links."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Finance Agent
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data and interpret trends.",
    model=Ollama(id="llama3.1"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=[
        "Retrieve stock prices, analyst recommendations, and key financial data.",
        "Focus on trends and present the data in tables with key insights."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Analyst Agent
analyst_agent = Agent(
    name="Analyst Agent",
    role="Ensure thoroughness and draw conclusions.",
    model=Ollama(id="llama3.1"),
    instructions=[
        "Check outputs for accuracy and completeness.",
        "Synthesize data to provide a final sentiment score (1-10) with justification."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Team of Agents
agent_team = Agent(
    model=Ollama(id="llama3.1"),
    team=[sentiment_agent, finance_agent, analyst_agent],
    instructions=[
        "Combine the expertise of all agents to provide a cohesive, well-supported response.",
        "Always include references and dates for all data points and sources.",
        "Present all data in structured tables for clarity.",
        "Explain the methodology used to arrive at the sentiment scores."
    ],
    show_tool_calls=True,
    markdown=True,
)

## Run Agent Team

# Final Prompt
def run_agent_team(companies):
    today = datetime.now().date()
    week_ago = today - timedelta(days=7)
    prompt = str(f"Analyze the sentiment for the following companies during the week of {week_ago} - {today}: {companies}. \n\n"
    "1. **Sentiment Analysis**: Search for relevant news articles and interpret th–e sentiment for each company. Provide sentiment scores on a scale of 1 to 10, explain your reasoning, and cite your sources.\n\n"
    "2. **Financial Data**: Analyze stock price movements, analyst recommendations, and any notable financial data. Highlight key trends or events, and present the data in tables.\n\n"
    "3. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-10) for each company. Justify the scores and provide a summary of the most important findings.\n\n"
    "Ensure your response is accurate, comprehensive, and includes references to sources with publication dates.")
    response: RunResponse = agent_team.run(prompt)
    return response

# print_run_response(run_agent_team(['NVDA', 'MSFT']), markdown=True)
