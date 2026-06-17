from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import requests
import random

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

def chat_node(state: ChatState):
    messages = [
        SystemMessage(
            content="Answer in plain text only. Do not use Markdown formatting, bold, italics, bullet points, or code blocks."
        )
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

#***************************************************Tools***********************************************

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """

    try:
        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}

            result = first_num / second_num

        else:
            return {
                "error": f"Unsupported operation '{operation}'. "
                         "Use add, sub, mul, or div."
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=JZ04KY5Z8PZY1J19"
    r = requests.get(url)
    
    data = r.json()["Global Quote"]

    return {
        "symbol": data["01. symbol"],
        "price": data["05. price"]
    }


# Make tool list
tools = [get_stock_price, search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)

conn = sqlite3.connect(database = 'chatbot.db',check_same_thread = False)
checkpointer = SqliteSaver(conn = conn)
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(
            checkpoint.config['configurable']['thread_id']
        )

    return list(all_threads)
