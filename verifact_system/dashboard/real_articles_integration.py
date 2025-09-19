"""
Integrazione articoli reali nel dashboard
"""
import json
import streamlit as st
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ingestion.news_fetcher import NewsArticle

@st.cache_data
def get_real_articles():
    """Carica articoli reali dal database"""
    articles_file = Path("data/real_articles.json")
    
    if not articles_file.exists():
        st.error("File articoli non trovato. Esegui prima: python scripts/populate_database.py")
        return []
    
    try:
        with open(articles_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = []
        # Prendi ultimi 50 articoli per performance
        for art_data in data.get('articles', [])[-50:]:
            article = NewsArticle(
                url=art_data['url'],
                title=art_data['title'],
                content=art_data['content'],
                author=art_data.get('author', 'Unknown'),
                published_date=datetime.fromisoformat(art_data['published_date']),
                source=art_data['source'],
                language=art_data.get('language', 'en'),
                quality_score=art_data.get('quality_score', 0.5)
            )
            articles.append(article)
        
        return articles
        
    except Exception as e:
        st.error(f"Errore caricamento articoli: {e}")
        return []