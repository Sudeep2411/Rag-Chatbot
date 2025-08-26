from typing import List
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer
import os
from app.src.utils.logger import get_logger

logger = get_logger("ChromaStore")

class ChromaStore:
    def __init__(self, persist_dir: str, embedding_model: str):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir, 
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name="rag_collection")
        self.embedder = SentenceTransformer(embedding_model)
        logger.info(f"Initialized ChromaStore with embedding model: {embedding_model}")

    def _embed(self, texts: List[str]):
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def add_texts(self, texts: List[str], metadatas: List[dict], ids: List[str]):
        if not texts:
            logger.warning("No texts to add to store")
            return
            
        logger.info(f"Adding {len(texts)} documents to vector store")
        embeddings = self._embed(texts)
        self.collection.add(
            documents=texts, 
            metadatas=metadatas, 
            ids=ids, 
            embeddings=embeddings
        )
        logger.info("Documents added successfully")

    def query(self, query: str, top_k: int = 4):
        logger.info(f"Querying store with: '{query[:50]}...'")
        q_emb = self._embed([query])[0]
        res = self.collection.query(
            query_embeddings=[q_emb], 
            n_results=top_k, 
            include=["distances", "metadatas", "documents", "embeddings"]
        )
        
        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
                "embedding": res["embeddings"][0][i] if res["embeddings"] else []
            })
        
        logger.info(f"Found {len(hits)} relevant documents")
        return hits

    def reset(self):
        logger.warning("Resetting Chroma database")
        self.client.reset()
