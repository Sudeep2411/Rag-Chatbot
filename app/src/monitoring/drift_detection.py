import numpy as np
from sklearn.decomposition import PCA
import json
import os
from datetime import datetime
from app.config import DRIFT_DETECTION_THRESHOLD
from app.src.utils.logger import get_logger

logger = get_logger("DriftDetection")

class DriftDetector:
    def __init__(self, persist_dir: str = "app/storage/monitoring"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.history_file = os.path.join(persist_dir, "embedding_history.json")
        self.history = self._load_history()
    
    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading history: {e}")
                return {"embeddings": [], "timestamps": []}
        return {"embeddings": [], "timestamps": []}
    
    def _save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def add_embeddings(self, embeddings: list):
        """Add new embeddings to history"""
        if not embeddings:
            return
            
        timestamp = datetime.now().isoformat()
        
        try:
            # Reduce dimensionality for storage
            if len(embeddings) > 0:
                # Convert to numpy array
                emb_array = np.array(embeddings)
                
                # Reduce dimensions if needed
                if emb_array.shape[1] > 3:
                    pca = PCA(n_components=3)
                    reduced = pca.fit_transform(emb_array)
                    centroids = np.mean(reduced, axis=0).tolist()
                else:
                    centroids = np.mean(emb_array, axis=0).tolist()
                
                self.history["embeddings"].append(centroids)
                self.history["timestamps"].append(timestamp)
                
                # Keep only last 1000 entries
                if len(self.history["embeddings"]) > 1000:
                    self.history["embeddings"] = self.history["embeddings"][-1000:]
                    self.history["timestamps"] = self.history["timestamps"][-1000:]
                
                self._save_history()
                logger.debug(f"Added {len(embeddings)} embeddings to history")
                
        except Exception as e:
            logger.error(f"Error adding embeddings to history: {e}")
    
    def detect_drift(self, new_embeddings: list) -> bool:
        """Detect if there's significant drift in the new embeddings"""
        if len(self.history["embeddings"]) < 10 or len(new_embeddings) == 0:
            return False  # Not enough data
        
        try:
            # Convert to numpy arrays
            historical_embs = np.array(self.history["embeddings"])
            new_embs = np.array(new_embeddings)
            
            # If we have high-dimensional data, reduce it
            if historical_embs.shape[1] > 3 or new_embs.shape[1] > 3:
                pca = PCA(n_components=3)
                all_embeddings = np.vstack([historical_embs, new_embs])
                reduced = pca.fit_transform(all_embeddings)
                
                # Split back into historical and new
                historical = reduced[:len(historical_embs)]
                new = reduced[len(historical_embs):]
            else:
                historical = historical_embs
                new = new_embs
            
            # Calculate centroids
            historical_centroid = np.mean(historical, axis=0)
            new_centroid = np.mean(new, axis=0)
            
            # Calculate distance between centroids
            distance = np.linalg.norm(historical_centroid - new_centroid)
            
            # Calculate spread of historical data
            historical_distances = np.linalg.norm(historical - historical_centroid, axis=1)
            max_historical_distance = np.max(historical_distances) if len(historical_distances) > 0 else 1.0
            
            # Normalize distance
            normalized_distance = distance / max_historical_distance if max_historical_distance > 0 else 0
            
            logger.debug(f"Drift detection - Distance: {normalized_distance:.3f}, Threshold: {DRIFT_DETECTION_THRESHOLD}")
            
            return normalized_distance > DRIFT_DETECTION_THRESHOLD
            
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return False
    
    def get_stats(self):
        """Get statistics about the embedding history"""
        if not self.history["embeddings"]:
            return {"count": 0, "latest_timestamp": None}
        
        return {
            "count": len(self.history["embeddings"]),
            "latest_timestamp": self.history["timestamps"][-1] if self.history["timestamps"] else None
        }
