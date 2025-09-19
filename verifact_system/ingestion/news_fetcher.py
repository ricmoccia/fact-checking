"""
VeriFact System - Simplified News Fetcher for Testing
Versione semplificata senza dipendenze asyncpg
"""

import hashlib
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

@dataclass
class NewsArticle:
    """Struttura dati per un articolo di news"""
    url: str
    title: str
    content: str
    author: Optional[str]
    published_date: datetime
    source: str
    language: str
    category: Optional[str] = None
    summary: Optional[str] = None
    quality_score: float = 0.0
    credibility_score: float = 0.0
    content_hash: Optional[str] = None
    
    def __post_init__(self):
        """Calcola hash contenuto per deduplicazione"""
        if self.content and not self.content_hash:
            content_text = f"{self.title}{self.content}"
            self.content_hash = hashlib.md5(
                content_text.encode('utf-8')
            ).hexdigest()

# Funzioni di utilità semplici per test
def create_test_article():
    """Crea articolo di test"""
    return NewsArticle(
        url="https://test.com/article",
        title="Test Article",
        content="This is test content for the fact-checking system.",
        author="Test Author",
        published_date=datetime.now(),
        source="Test Source",
        language="en"
    )