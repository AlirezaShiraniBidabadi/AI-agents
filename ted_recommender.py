import pandas as pd
import numpy as np
import kagglehub 
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate ,ChatPromptTemplate
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
import tiktoken
import faiss
import openai 
from openai import OpenAI
from langchain_openai import ChatOpenAI
import os
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph 
from langchain_core.tools import tool
checkpointer = InMemorySaver()
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = 'open_ai_key'
client = OpenAI() 
llm = ChatOpenAI(model_name="gpt-4o")
# Download latest version
path = kagglehub.dataset_download("rounakbanik/ted-talks")

df =  pd.read_csv(path + '/ted_main.csv')
df1 = pd.read_csv(path + '/transcripts.csv')

df_merged = df.merge(df1,on='url')

# Defining the system prompt (how the AI should act)
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that helps generate transcription summaries."
)
# the user prompt is provided by the user, in this case however the only dynamic
# input is the article
user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a very short summary (6 sentences max) for an video transcription.
The transcription is here for you to examine {transcription}

The summary should be based of the context of the trascription.

Only output the transcription summary, no other explanation or
text is needed.""",
    input_variables=["transcription"]
)

first_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
class State(MessagesState):
    context: str
def summarizer(state:State):
  transcription = state['context']
  # Get context if it exists
  prompt = first_prompt.format(transcription=transcription)
  response = llm.invoke(prompt)
  dict1 =  {'messages':response}
  return dict1

graph = StateGraph(State)
graph.add_node("summarizer", summarizer)
graph.add_edge(START, "summarizer")
graph.add_edge("summarizer", END)
# Compile the graph
d_graph = graph.compile()

def len_string(string):
  return len(string)
df_merged['len_transcript'] = df_merged['transcript'].apply(len_string)
df_merged = df_merged.sort_values(by='len_transcript')
df_t = df_merged.iloc[10:100,:]

#Initialize the encoding for the specific model you're using
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def token_count(text):
   return len(encoding.encode(text))
df_t['token_count'] = df_t['transcript'].apply(token_count)


#extract all summaries and store them in a new column in the original dataframe
summaries = []
for row in range(df_t['transcript'].shape[0]):
    response = d_graph.invoke({'context': df_t['transcript'].values[row]})['messages']
    summaries.append(response[-1].content)

sumcol = np.array(summaries)
df_t['summary'] = sumcol
df_t['talk_code'] = 'speaker_name_and_the_subject: ' + df_t['name'] + ' talk_description: ' + df_t['description'] + ' talk_url: ' + df_t['url']
df_t['completed_talk_code'] =  df_t['talk_code'] + ' talk_summary: ' + df_t['summary'] 
# Encode function using OpenAI embedding API
def get_openai_embedding(text):
    response =  client.embeddings.create(input=text, model="text-embedding-3-small")
    return np.array(response.data[0].embedding, dtype=np.float32)
# Example DataFrame summaries encoding
embeddings = np.vstack([get_openai_embedding(text) for text in df_t['summary'].values])

# Build FAISS index
dim = embeddings.shape[1] 
index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(embeddings)
index.add(embeddings)
# Query embedding and search similar....here is whre recommander starts
@tool
def similarity_searcher(query="a short talk about solar energy",k=5): 
    """
    searches through ted_talk summaries and returns top k most relevant ted talks to the user's query.

    Args:
        query (str): The user's query to search among ted talks.
        k (int) : determines how many ted talks shoud be retuended to the user
        
    Returns:
        dict: a list of ted talks related to the user's query. 
    """
    query_vec = get_openai_embedding(query).reshape(1, -1)
    faiss.normalize_L2(query_vec)
    k = k
    distances, indices = index.search(query_vec, k)
    # print(distances)
    # print(indices)
    return df_t.iloc[indices[0],:]['completed_talk_code'].values 

# Bind the tool-enabled LLM with the web_search tool
llm_with_tools = llm.bind_tools([similarity_searcher])


# Define a State subclass to hold the relevant attributes for agent state management
class State(MessagesState):
    summaries: list[str]
    query: str


def agent(state: State) -> dict:
    """
    Agent function that processes the state and generates responses using LLM with or without context.

    Args:
        state (State): The current state containing messages, results, and query info.

    Returns:
        dict: A dictionary containing the new messages and optionally the query for further processing.
    """
    human_input = state['messages']
    # Retrieve existing context results if available, default empty string
    ted_talks = state.get("summaries", "")

    # If context results exist, add them to the system message for enhanced answer generation
    if ted_talks:
        system_message = (
            f"Here is a quey provided by the user about a ted talk. "
            f"You should reccommend ted talks to the user using context in the following, that includes the related ted talks to what user has asked. "
            f"Here is the context: {ted_talks}"
        )
        messages = [SystemMessage(content=system_message)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {'messages': response}

    # Otherwise, instruct the LLM to call the web_search tool to gather relevant context
    else:
        system_message = (
            "Here is a quey provided by the user about a ted talk. You should call the similarity_searcher tool "
            "to provide relatable context for you in relation to the user's query. "
            "Rewrite a proper question in order to search for relevant docs. "
            "Do not say 'here is your question', just write the question itself."
        )
        messages = [SystemMessage(content=system_message)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {'messages': response, 'query': response.content}


# Function to create the state graph which integrates the agent and web search tool node
def ted_recommender_agent() -> StateGraph:
    """
    Constructs a StateGraph for the web searching agent workflow.

    Returns:
        StateGraph: Configured graph with nodes and edges for agent operation.
    """
    builder = StateGraph(MessagesState)
    builder.add_node("search_agent", agent)
    builder.add_node("tools", ToolNode([similarity_searcher]))
    builder.add_edge(START, "search_agent")

    # Conditional routing based on whether the latest assistant message is a tool call
    builder.add_conditional_edges(
        "search_agent",
        tools_condition,
    )
    builder.add_edge("tools", "search_agent")
    graph = builder.compile(checkpointer=checkpointer)
    return graph
