"""
Ingestion automatica con NewsAPI
"""
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from newsapi import NewsApiClient

# Sostituisci con la tua API key
NEWSAPI_KEY = "192c3f7e7147494c80120fd128292d18"

class AutoNewsIngestion:
    def __init__(self, api_key):
        self.newsapi = NewsApiClient(api_key=api_key)
        self.articles_file = Path("data/real_articles.json")
        
    def load_existing_articles(self):
        """Carica articoli esistenti"""
        if self.articles_file.exists():
            with open(self.articles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('articles', [])
        return []
    
    def fetch_latest_news(self, topics=['fact check', 'climate change', 'covid vaccine']):
        """Fetch da NewsAPI"""
        all_articles = []
        
        for topic in topics:
            try:
                articles = self.newsapi.get_everything(
                    q=topic,
                    language='en',
                    sort_by='publishedAt',
                    from_param=(datetime.now() - timedelta(days=1)).isoformat(),
                    page_size=50
                )
                
                for article in articles['articles']:
                    processed = {
                        'url': article['url'],
                        'title': article['title'],
                        'content': article['description'] or article['content'] or '',
                        'author': article['author'] or 'Unknown',
                        'published_date': article['publishedAt'],
                        'source': article['source']['name'],
                        'language': 'en',
                        'quality_score': 0.7,
                        'topic': topic
                    }
                    all_articles.append(processed)
                    
                print(f"Trovati {len(articles['articles'])} articoli per '{topic}'")
                
            except Exception as e:
                print(f"Errore topic {topic}: {e}")
        
        return all_articles
    
    def save_articles(self, new_articles):
        """Salva articoli aggiornati"""
        existing = self.load_existing_articles()
        existing_urls = {art['url'] for art in existing}
        
        # Aggiungi solo nuovi articoli
        added = 0
        for article in new_articles:
            if article['url'] not in existing_urls:
                existing.append(article)
                added += 1
        
        # Salva
        data = {
            'last_update': datetime.now().isoformat(),
            'total_articles': len(existing),
            'articles': existing
        }
        
        with open(self.articles_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Aggiunti {added} nuovi articoli. Totale: {len(existing)}")

def main():
    if NEWSAPI_KEY == "YOUR_API_KEY_HERE":
        print("ERRORE: Sostituisci YOUR_API_KEY_HERE con la tua vera API key!")
        return
    
    ingestion = AutoNewsIngestion(NEWSAPI_KEY)
    new_articles = ingestion.fetch_latest_news()
    ingestion.save_articles(new_articles)

if __name__ == "__main__":
    main()