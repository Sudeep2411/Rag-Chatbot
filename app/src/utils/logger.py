import logging
import os
from app.config import LOG_LEVEL

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        
        # Set log level from config
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)
        
        fmt = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    
    return logger