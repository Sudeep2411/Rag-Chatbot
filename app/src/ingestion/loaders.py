from typing import List, Dict
from pypdf import PdfReader
import os, re, requests
from bs4 import BeautifulSoup
from app.src.utils.logger import get_logger

logger = get_logger("Loaders")

def load_txt(path: str) -> List[Dict]:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return [{
            'content': text,
            'metadata': {'source': os.path.basename(path), 'type': 'txt', 'path': path}
        }]
    except Exception as e:
        logger.error(f"Error loading TXT file {path}: {e}")
        return []

def load_pdf(path: str) -> List[Dict]:
    try:
        reader = PdfReader(path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():  # Only add non-empty pages
                docs.append({
                    'content': text,
                    'metadata': {'source': os.path.basename(path), 'type': 'pdf', 'page': i+1, 'path': path}
                })
        logger.info(f"Loaded {len(docs)} pages from PDF: {os.path.basename(path)}")
        return docs
    except Exception as e:
        logger.error(f"Error loading PDF file {path}: {e}")
        return []

def load_url(url: str) -> List[Dict]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.get(url, timeout=30, headers=headers)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Remove unwanted elements
        for t in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav']):
            t.extract()
        
        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.body
        text = main_content.get_text(separator=' ', strip=True) if main_content else ''
        text = re.sub(r'\s+', ' ', text).strip()
        
        title = soup.title.string.strip() if soup.title else url
        
        logger.info(f"Loaded URL: {title}")
        return [{
            'content': text,
            'metadata': {'source': title, 'type': 'url', 'url': url}
        }]
    except Exception as e:
        logger.error(f"Error loading URL {url}: {e}")
        return []

def discover_and_load(data_dir: str) -> List[Dict]:
    docs = []
    
    # Load local files
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for fn in files:
                path = os.path.join(root, fn)
                if fn.lower().endswith('.pdf'):
                    docs.extend(load_pdf(path))
                elif fn.lower().endswith('.txt'):
                    docs.extend(load_txt(path))
    
    # Load URLs from urls.txt
    urls_path = os.path.join(data_dir, 'urls.txt')
    if os.path.exists(urls_path):
        with open(urls_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'):  # Skip comments
                    try:
                        docs.extend(load_url(url))
                    except Exception as e:
                        logger.warning(f"Failed to load URL {url}: {e}")
    
    logger.info(f"Total documents loaded: {len(docs)}")
    return docs
