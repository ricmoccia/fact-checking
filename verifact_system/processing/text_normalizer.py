"""
VeriFact System - Basic Text Normalizer
Versione semplificata per test iniziali
"""

import re
from typing import Optional

def normalize_text(text: str, language: Optional[str] = None) -> str:
    """Normalizzazione base del testo"""
    if not text:
        return ""
    
    # Pulizia base
    # Rimuovi caratteri speciali eccessive
    text = re.sub(r'\s+', ' ', text)  # Spazi multipli -> singolo
    text = re.sub(r'\n+', ' ', text)  # Newlines -> spazio
    
    # Rimuovi HTML tags base
    text = re.sub(r'<[^>]+>', '', text)
    
    # Trim
    text = text.strip()
    
    return text

def assess_text_quality(text: str) -> float:
    """Valutazione base qualità testo"""
    if not text:
        return 0.0
    
    score = 1.0
    
    # Penalizza testi troppo corti
    if len(text) < 100:
        score -= 0.3
    
    # Penalizza testi con troppe ripetizioni
    if len(set(text.split())) < len(text.split()) * 0.5:
        score -= 0.2
    
    return max(0.0, score)