from typing import List, Dict
from app.src.utils.logger import get_logger

logger = get_logger("PostProcess")

def to_citations(hits: List[Dict]) -> List[Dict]:
    enriched = []
    for h in hits:
        dist = h.get('distance', 1.0) or 1.0
        conf = 1.0/(1.0+dist)
        meta = h.get('metadata', {})
        source = meta.get('source') or meta.get('url') or 'source'
        
        if meta.get('type') == 'pdf' and meta.get('page'):
            source = f"{source} p.{meta['page']}"
        
        enriched.append({
            'text': h.get('document', ''),
            'source': source,
            'confidence': round(conf, 3),
            'metadata': meta
        })
    
    logger.debug(f"Processed {len(enriched)} citations")
    return enriched

def format_sources(citations: List[Dict]) -> str:
    if not citations:
        return "No sources found"
    
    lines = []
    for i, c in enumerate(citations, 1):
        lines.append(f"[{i}] {c['source']} (confidence {c['confidence']})")
    
    return "\n".join(lines)
