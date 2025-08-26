from typing import Dict, List
from app.src.utils.store import ChromaStore
from app.src.utils.prompts import build_prompt, SYSTEM_PROMPT
from app.src.utils.postprocess import to_citations, format_sources
from app.src.models.generator import TextGenerator
from app.config import PERSIST_DIR, EMBEDDING_MODEL, TOP_K
from app.src.utils.logger import get_logger
from app.src.monitoring.drift_detection import DriftDetector
from app.src.monitoring.feedback import FeedbackLogger

logger = get_logger("RAG")

class RAGPipeline:
    def __init__(self, persist_dir: str = PERSIST_DIR, embedding_model: str = EMBEDDING_MODEL):
        self.store = ChromaStore(persist_dir=persist_dir, embedding_model=embedding_model)
        self.generator = TextGenerator()
        self.drift_detector = DriftDetector()
        self.feedback_logger = FeedbackLogger()
        logger.info("RAGPipeline initialized")

    def retrieve(self, query: str, top_k: int = TOP_K):
        return self.store.query(query, top_k=top_k)

    def answer(self, question: str, top_k: int = TOP_K) -> Dict:
        logger.info(f"Processing question: '{question[:50]}...'")
        
        # Retrieve relevant documents
        hits = self.retrieve(question, top_k=top_k)
        citations = to_citations(hits)

        # Monitor for concept drift
        embeddings = [hit.get('embedding', []) for hit in hits if 'embedding' in hit]
        if embeddings:
            drift_detected = self.drift_detector.detect_drift(embeddings)
            if drift_detected:
                logger.warning("Concept drift detected in retrieved embeddings")

        # Prepare context for prompt
        context_items = [{
            'text': c['text'],
            'source': c['source']
        } for c in citations]

        # Build and execute prompt
        prompt = f"{SYSTEM_PROMPT}\n\n" + build_prompt(question, context_items)
        logger.debug(f"Generated prompt length: {len(prompt)}")
        
        generated = self.generator.generate(prompt)
        sources_block = format_sources(citations)
        
        # Log the interaction
        self.feedback_logger.log_feedback(
            question=question,
            answer=generated,
            sources=citations,
            metadata={"top_k": top_k}
        )
        
        logger.info("Question processed successfully")
        return {
            'answer': generated,
            'sources': citations,
            'sources_text': sources_block
        }
    
    def log_feedback(self, question: str, answer: str, sources: list, rating: int = None, 
                    user_feedback: str = None):
        """Log user feedback for quality improvement"""
        self.feedback_logger.log_feedback(
            question=question,
            answer=answer,
            sources=sources,
            rating=rating,
            user_feedback=user_feedback
        )
