import os
from dotenv import load_dotenv
import logging
from langchain_core.documents import Document

# LangChain imports for embeddings and vector store
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

# Set up basic logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- 1. Environment and Configuration ---
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_URL = os.getenv("OLLAMA_URL")
COLLECTION_NAME = "my_personal_knowledge"

# --- 2. Define the Embedding Model and Vector Store ---
try:
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_URL)
    vector_store = PGVector(
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        embeddings=embeddings,
        use_jsonb=True,
    )
except Exception as e:
    log.error(f"Failed to connect to the database or Ollama: {e}")
    exit(1)

# --- 3. Your Personal Data ---
# This is where you would put your personal information.
# Each entry is a 'Document' with content and optional metadata.
personal_data = [
    Document(
        page_content="I was born in a small town called Spring Valley, in the state of Illinois.",
        metadata={"category": "places", "type": "lived"},
    ),
    Document(
        page_content="I worked for a company named Google as a software engineer from 2015 to 2020.",
        metadata={"category": "work", "company": "Google"},
    ),
    Document(
        page_content="After Google, I joined a tech startup called OpenAI in 2020.",
        metadata={"category": "work", "company": "OpenAI"},
    ),
    Document(
        page_content="In 2022, I took a trip to Japan and visited Tokyo, Kyoto, and Osaka.",
        metadata={"caategory": "places", "type": "visited"},
    ),
    Document(
        page_content="My first apartment was in Chicago, where I lived for five years.",
        metadata={"category": "places", "type": "lived"},
    ),
]

# --- 4. The Ingestion Process ---
if __name__ == "__main__":
    log.info("Starting data ingestion into the PGVector store...")

    try:
        # Add the documents. This will handle chunking, embedding, and saving.
        vector_store.add_documents(documents=personal_data)
        log.info("Data ingestion complete!")

        # Verify by performing a simple search
        log.info("Verifying with a sample query...")
        retriever = vector_store.as_retriever()
        results = retriever.invoke("places I have lived in")

        log.info("Retrieved results:")
        for doc in results:
            log.info(f"- Content: '{doc.page_content}'")
            log.info(f"  Metadata: {doc.metadata}")

    except Exception as e:
        log.error(f"An error occurred during ingestion or verification: {e}")
