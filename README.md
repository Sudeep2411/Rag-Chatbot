# 🤖 RAG Chatbot with Llama 3 Integration

A sophisticated **Retrieval-Augmented Generation (RAG) chatbot** that leverages **Llama 3 via Ollama** for intelligent **document-based question answering**.  
This application combines **document intelligence** with **state-of-the-art language models** to provide **accurate, source-cited responses**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red)
![Ollama](https://img.shields.io/badge/Ollama-Llama%203-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-green)

---

## 🚀 Features

| Category              | Features                                                                 |
|------------------------|--------------------------------------------------------------------------|
| **Document Processing** | PDF/TXT/URL ingestion, Text chunking, Sentence embeddings               |
| **AI Capabilities**     | Llama 3 integration, Context-aware responses, Source citation           |
| **Vector Database**     | ChromaDB storage, Semantic search, Embedding management                 |
| **User Interface**      | Streamlit web app, File upload, Real-time chat                          |
| **Monitoring**          | Feedback system, Concept drift detection, Performance analytics         |
| **Deployment**          | FastAPI backend, Docker-ready, Hugging Face Spaces compatible           |

---

## 📋 Prerequisites

| Requirement | Version | Installation Command |
|-------------|---------|-----------------------|
| **Python**  | 3.9+    | [Download](https://python.org) |
| **Ollama**  | Latest  | `curl -fsSL https://ollama.ai/install.sh \| sh` |
| **Llama 3** | 8B/70B  | `ollama pull llama3` |

---

## 🛠️ Installation

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd rag-chatbot
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Ollama & Llama 3
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3 model
ollama pull llama3

# Start Ollama service (in separate terminal)
ollama serve
```

### 4. Environment Configuration
Create `.env` file:
```bash
# Model configurations
MODEL_ID=llama3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
USE_OLLAMA=True

# Processing configurations
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=4

# Storage configurations
PERSIST_DIR=app/storage/chroma
DATA_DIR=app/data

# Monitoring configurations
DRIFT_DETECTION_THRESHOLD=0.7
LOG_LEVEL=INFO
```

---

## 📁 Project Structure

```
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── app.py                 # Streamlit frontend
│   ├── api.py                 # FastAPI backend
│   ├── config.py              # Configuration settings
│   ├── data/                  # Document storage
│   │   ├── urls.txt           # URLs to scrape
│   │   └── sample.txt         # Example document
│   └── src/
│       ├── ingestion/         # Document processing
│       │   ├── ingest.py      # Main ingestion script
│       │   └── loaders.py     # Document loaders
│       ├── utils/             # Core utilities
│       │   ├── store.py       # ChromaDB wrapper
│       │   ├── prompts.py     # Prompt templates
│       │   ├── postprocess.py # Result formatting
│       │   ├── logger.py      # Logging setup
│       │   └── rag.py         # Main RAG pipeline
│       ├── models/            # Model handling
│       │   └── generator.py   # Text generation
│       ├── monitoring/        # Monitoring system
│       │   ├── drift_detection.py
│       │   └── feedback.py    # User feedback
│       ├── finetuning/        # Fine-tuning
│       │   └── lora.py        # LoRA implementation
│       └── backend/           # Backend integration
│           └── langchain_wrapper.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Usage

| Step | Action |
|------|--------|
| **1. Add Documents** | Place files in `app/data/` (PDF/TXT) or add URLs in `urls.txt`. |
| **2. Ingest Documents** | `python -m app.src.ingestion.ingest` |
| **3. Start App (Streamlit)** | `streamlit run app/app.py` |
| **4. Or Start Backend (FastAPI)** | `python -m app.api` |
| **5. Access Interfaces** | Web UI: `http://localhost:8501` <br> API Docs: `http://localhost:8000/docs` |

---

## ⚙️ Configuration Options

| Setting         | Default | Description |
|-----------------|---------|-------------|
| MODEL_ID        | llama3  | Primary language model |
| OLLAMA_MODEL    | llama3  | Ollama model name |
| USE_OLLAMA      | True    | Use Ollama instead of Hugging Face |
| CHUNK_SIZE      | 800     | Text chunk size |
| CHUNK_OVERLAP   | 120     | Overlap between chunks |
| TOP_K           | 4       | Number of chunks retrieved |

---

## 🔧 API Endpoints

| Endpoint     | Method | Description |
|--------------|--------|-------------|
| `/`          | GET    | API info |
| `/health`    | GET    | System health check |
| `/ask`       | POST   | Query the RAG system |
| `/feedback`  | POST   | Submit feedback |
| `/stats`     | GET    | System statistics |

---

## 📊 Monitoring & Analytics

- **Feedback Logging** → Tracks user ratings & comments  
- **Concept Drift Detection** → Monitors embedding drift  
- **Performance Metrics** → Response accuracy & latency  
- **Usage Statistics** → Query volume and user patterns  

---

## 🐳 Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

EXPOSE 8501 8000

CMD ["sh", "-c", "ollama serve & streamlit run app/app.py"]
```

---

## 🌐 Hugging Face Spaces Deployment

```yaml
# huggingface.yaml
image: python:3.9
variables:
  HF_TOKEN: $HF_TOKEN
install:
  - pip install -r requirements.txt
  - curl -fsSL https://ollama.ai/install.sh | sh
  - ollama pull llama3
script:
  - ollama serve &
  - streamlit run app/app.py --server.port $PORT
```

---

## 🔍 Troubleshooting

| Issue                     | Solution |
|----------------------------|----------|
| **Ollama connection failed** | Ensure `ollama serve` is running |
| **Model not found**        | Run `ollama pull llama3` |
| **Port conflicts**         | Change ports in `.env` |
| **Memory issues**          | Reduce chunk size or use smaller model |

**Performance Optimization**  
- CHUNK_SIZE: `600–1000` chars  
- TOP_K: `3–5` chunks  
- Adjust Ollama params based on available RAM  

---

## 🤝 Contributing

1. Fork the repo  
2. Create a branch (`git checkout -b feature/amazing-feature`)  
3. Commit (`git commit -m 'Add feature'`)  
4. Push (`git push origin feature/amazing-feature`)  
5. Open a Pull Request  

---

## 📄 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments
- **Meta AI** → Llama 3 model  
- **Ollama** → Local model serving  
- **ChromaDB** → Vector database  
- **Hugging Face** → Transformer models  
- **Streamlit** → Web interface  

---

