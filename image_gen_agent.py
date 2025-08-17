# author : alireza shirani
# Description: 
# This code uses the OpenAI API to generate an image based on the given query 
# It returns an url refering to the generated image.
#
# Usage example:
# from image_agent import image_agent
# d_graph_image_agent = image_agent()
# d_graph_image_agent.invoke({'messages':['generate a pic about a ted talk in which people are talking about solar energy.']})['image_url']
 

#importing libraries
from langchain_core.messages import SystemMessage, HumanMessage 
from langgraph.graph import MessagesState
from typing import Annotated
from langgraph.graph.message import add_messages
from openai import OpenAI 
from IPython.display import Image 
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI 
import os 
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# Ensure OpenAI API key is set as environment variable, prompt or set default if missing
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = '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'

# Initialize the ChatOpenAI LLM model with the specified model_name
llm = ChatOpenAI(model_name="gpt-4o")

client = OpenAI()

checkpointer = InMemorySaver()


class State(MessagesState):
    """
    State class extending MessagesState to hold the current state of the agent.
    
    Attributes:
        image_url (str): URL of the generated image.
    """
    image_url: str
    query: str
    
@tool
def image_generator(query: str)-> dict:
    """
    image_generator function that processes a user's query,
    and generates an image using the DALL-E 3 model.
    
    Args:
        query (str): the input query to te image generator model
        
    Returns:
        image_url (str): The URL of the generated image.    """

    result = client.images.generate(model="dall-e-3",
                                   prompt=query, size="1024x1024")
    image_url = result.data[0].url
    return image_url

llm_with_tools = llm.bind_tools([image_generator])

def agent(state: State) -> dict:
    """
    Agent function that processes the state and generates responses using LLM with or without context.

    Args:
        state (State): The current state containing messages, results, and query info.

    Returns:
        dict: A dictionary containing the new messages and optionally the query for further processing.
    """
    # Retrieve existing context results if available, default empty string
    url = state.get("image_url", "")

    # If context results exist, add them to the system message for enhanced answer generation
    if url:
        print(1)
        system_message = (
            f"Here is a query for image generation provided by the user. "
            f"and also here is the generated image url {url} answer to the user and print the url of te generated image"
        )
        messages = [SystemMessage(content=system_message)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {'messages': response}

    # Otherwise, instruct the LLM to call the web_search tool to gather relevant context
    else:
        system_message = (
            "Here is a query for image generation provided by the user. You should call the image_generator tool "
            "to provide generated image's url for you in relation to the user's query. "
            "Rewrite a proper query in order to generate the right image for user. " 
        )
        messages = [SystemMessage(content=system_message)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {'messages': response, 'query': response.content}


def image_agent():
    """
    Constructs and compiles a state graph for the image generation agent.
    
    The graph includes:
        - A start node leading to the image_agent node running the agent function.
        - An edge from the image_agent node to the end node.
    
    Returns:
        Compiled state graph object ready for invocation.
    """
    builder = StateGraph(MessagesState)
    builder.add_node("image_agent", agent)
    builder.add_node("tools", ToolNode([image_generator]))
    builder.add_edge(START, "image_agent")

    # Conditional routing based on whether the latest assistant message is a tool call
    builder.add_conditional_edges(
        "image_agent",
        tools_condition,
    )
    builder.add_edge("tools", "image_agent")
    d_graph = builder.compile(checkpointer=checkpointer)
    return d_graph
