import os, uuid
from typing import List
from app.src.ingestion.loaders import discover_and_load
from app.src.utils.store import ChromaStore
from app.config import PERSIST_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR
from app.src.utils.logger import get_logger

logger = get_logger("INGEST")

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(start + chunk_size, n)
        # Try to split at sentence boundary
        if end < n:
            # Look for sentence endings near the chunk end
            for split_point in range(end, start + chunk_size - overlap, -1):
                if split_point < n and text[split_point] in '.!?。！？\n':
                    end = split_point + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        start = end - overlap
        if start < 0:
            start = 0
        if end >= n:
            break
    
    return chunks

def main():
    logger.info(f"Scanning {DATA_DIR} for documents...")
    docs = discover_and_load(DATA_DIR)
    logger.info(f"Loaded {len(docs)} raw documents (pre-chunking).")    

    store = ChromaStore(persist_dir=PERSIST_DIR, embedding_model=EMBEDDING_MODEL)

    texts, metas, ids = [], [], []
    for d in docs:
        chunks = split_text(d['content'], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            meta = dict(d['metadata'])
            meta['chunk'] = i + 1
            meta['total_chunks'] = len(chunks)
            metas.append(meta)
            ids.append(str(uuid.uuid4()))
    
    logger.info(f"Prepared {len(texts)} chunks. Embedding & adding to store...")

    if texts:
        store.add_texts(texts, metas, ids)
        logger.info("Ingestion complete. Chroma persisted.")
    else:
        logger.warning("No text found to ingest. Put PDFs/TXTs in app/data or URLs in app/data/urls.txt.")

if __name__ == "__main__":
    main()
