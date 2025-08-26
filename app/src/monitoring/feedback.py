import sqlite3
import json
from datetime import datetime
import os
from app.src.utils.logger import get_logger

logger = get_logger("Feedback")

class FeedbackLogger:
    def __init__(self, db_path: str = "app/storage/feedback.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    rating INTEGER,
                    user_feedback TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create index for better query performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_rating ON feedback(rating)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Feedback database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing feedback database: {e}")
    
    def log_feedback(self, question: str, answer: str, sources: list, rating: int = None, 
                    user_feedback: str = None, metadata: dict = None):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback (timestamp, question, answer, sources, rating, user_feedback, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                question[:500],  # Limit length to prevent DB issues
                answer[:1000],
                json.dumps(sources),
                rating,
                user_feedback[:500] if user_feedback else None,
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            logger.debug("Feedback logged successfully")
            
        except Exception as e:
            logger.error(f"Error logging feedback: {e}")
    
    def get_feedback_stats(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total count and average rating
            cursor.execute('SELECT COUNT(*), AVG(rating) FROM feedback WHERE rating IS NOT NULL')
            count_result = cursor.fetchone()
            total_count = count_result[0] if count_result else 0
            avg_rating = float(count_result[1]) if count_result and count_result[1] is not None else 0.0
            
            # Get feedback with comments count
            cursor.execute('SELECT COUNT(*) FROM feedback WHERE user_feedback IS NOT NULL AND user_feedback != ""')
            feedback_count = cursor.fetchone()[0]
            
            # Get rating distribution
            cursor.execute('''
                SELECT rating, COUNT(*) 
                FROM feedback 
                WHERE rating IS NOT NULL 
                GROUP BY rating 
                ORDER BY rating
            ''')
            rating_distribution = {str(row[0]): row[1] for row in cursor.fetchall()}
            
            # Get recent feedback
            cursor.execute('''
                SELECT timestamp, question, rating, user_feedback 
                FROM feedback 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            recent_feedback = [
                {
                    "timestamp": row[0],
                    "question": row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                    "rating": row[2],
                    "feedback": row[3]
                }
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
            return {
                "total_feedback": total_count,
                "average_rating": round(avg_rating, 2),
                "text_feedback_count": feedback_count,
                "rating_distribution": rating_distribution,
                "recent_feedback": recent_feedback
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {
                "total_feedback": 0,
                "average_rating": 0,
                "text_feedback_count": 0,
                "rating_distribution": {},
                "recent_feedback": []
            }
    
    def export_feedback(self, output_path: str):
        """Export all feedback to a JSON file"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM feedback ORDER BY timestamp')
            columns = [description[0] for description in cursor.description]
            feedback_data = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for item in feedback_data:
                if item.get('sources'):
                    item['sources'] = json.loads(item['sources'])
                if item.get('metadata'):
                    item['metadata'] = json.loads(item['metadata'])
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, indent=2, ensure_ascii=False)
            
            conn.close()
            logger.info(f"Feedback exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}")
            return False
