import os
from dotenv import load_dotenv

load_dotenv()

# Model configurations
MODEL_ID = os.getenv("MODEL_ID", "google/flan-t5-base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")  # Changed default to llama3
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
USE_OLLAMA = os.getenv("USE_OLLAMA", "False").lower() == "true"

# Processing configurations
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
TOP_K = int(os.getenv("TOP_K", 4))

# Storage configurations
PERSIST_DIR = os.getenv("PERSIST_DIR", "app/storage/chroma")
DATA_DIR = os.getenv("DATA_DIR", "app/data")

# API configurations
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Monitoring configurations
DRIFT_DETECTION_THRESHOLD = float(os.getenv("DRIFT_DETECTION_THRESHOLD", 0.7))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Deployment configurations
HF_REPO_ID = os.getenv("HF_REPO_ID", "")
