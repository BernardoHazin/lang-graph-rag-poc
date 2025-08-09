import os
from dotenv import load_dotenv
from typing import TypedDict, List

# LangGraph and LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres.vectorstores import PGVector
from langgraph.graph import StateGraph, END

# --- 1. Environment Setup ---
# Load environment variables from .env file
load_dotenv()

# --- 2. Database and Retrieval Tool Setup ---
DATABASE_URL = os.getenv("DATABASE_URL")

OLLAMA_URL = os.getenv("OLLAMA_URL")

# --- 2. Database and Retrieval Tool Setup ---
# You need to have a table with text and vector columns.
# Let's assume you have a table named 'documents' with 'content' and 'embedding' columns.
COLLECTION_NAME = "my_personal_knowledge"

# Define the embedding model. You'll need to have 'ollama pull' this model.
# e.g., ollama pull mxbai-embed-large
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_URL)

# Connect to the PGVector store
# This assumes your database is already running via docker-compose
vector_store = PGVector(
    connection=DATABASE_URL,
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    use_jsonb=True 
)

# Create a retriever from the vector store
retriever = vector_store.as_retriever()

# --- 3. Language Model (LLM) Setup ---
# Define the LLM model. This assumes 'ollama pull llama3' has been run.
llm = ChatOllama(model="llama3", base_url=OLLAMA_URL)

# --- 4. LangGraph Agent Definition ---

# Define a custom state for our graph.
# It will hold the user's message and the retrieved context.
class AgentState(TypedDict):
    """Represents the state of our graph."""
    input: str
    context: str
    chat_history: List[BaseMessage]
    response: str

# Define the nodes of our graph.
def retrieve_node(state: AgentState):
    """
    Retrieves relevant documents from the vector store based on the user's input.
    """
    print("---RETRIEVING CONTEXT---")
    query = state["input"]
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"context": context}

def generate_node(state: AgentState):
    """
    Generates a response using the LLM and the retrieved context.
    """
    print("---GENERATING RESPONSE---")
    
    # Create the RAG prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant that answers questions based ONLY on the provided context."),
            ("system", "Context: {context}"),
            ("human", "{input}")
        ]
    )

    # Create the generation chain
    rag_chain = (
        RunnablePassthrough() 
        | prompt_template 
        | llm 
        | StrOutputParser()
    )
    
    # Generate the response
    response = rag_chain.invoke({
        "context": state["context"],
        "input": state["input"]
    })
    
    return {"response": response}

# Build the LangGraph
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# Define the edges
workflow.add_edge("retrieve", "generate")

# Set the entry point
workflow.set_entry_point("retrieve")

# Set the end point
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()

# --- 5. Usage Example ---
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Before running, you must:
    # 1. Run 'docker-compose up' to start the database.
    # 2. Run 'ollama serve' and 'ollama pull llama3' and 'ollama pull mxbai-embed-large'
    # 3. Use a script to populate your PGVector table with some personal data.

    # Example question
    question = "What companies have I worked for?"
    
    # Run the graph with a question
    # The graph will retrieve context, then generate a response
    final_state = app.invoke({"input": question})

    print("\n--- RESPONSE ---")
    print(final_state["response"])

    final_state = app.invoke({"input": "What is my name?"})

    print("\n--- RESPONSE ---")
    print(final_state["response"])

    final_state = app.invoke({"input": "What places have I visited?"})

    print("\n--- RESPONSE ---")
    print(final_state["response"])
