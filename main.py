import os
from dotenv import load_dotenv
from typing import TypedDict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

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

# --- 2. FastAPI Setup ---
app = FastAPI(
    title="My Life Chat API",
    description="A simple API to interact with your personal knowledge base using RAG",
    version="1.0.0"
)

# Pydantic models for API requests and responses
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    context: str = None

# --- 3. Database and Retrieval Tool Setup ---
DATABASE_URL = os.getenv("DATABASE_URL")

OLLAMA_URL = os.getenv("OLLAMA_URL")

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

# --- 4. Language Model (LLM) Setup ---
# Define the LLM model. This assumes 'ollama pull llama3' has been run.
llm = ChatOllama(model="llama3", base_url=OLLAMA_URL)

# --- 5. LangGraph Agent Definition ---

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
rag_workflow = workflow.compile()

# --- 6. API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Send a message to the AI assistant and get a response based on your personal knowledge base.
    """
    try:
        # Run the RAG workflow with the user's message
        final_state = rag_workflow.invoke({"input": request.message})
        
        return ChatResponse(
            response=final_state["response"],
            context=final_state.get("context", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy", "message": "My Life Chat API is running"}

# --- 7. Usage Example ---
if __name__ == "__main__":
    print("Starting My Life Chat API server...")
    print("API will be available at: http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    print("Chat endpoint: POST http://localhost:8000/chat")
    print("Press Ctrl+C to stop the server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
