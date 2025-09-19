"""
Popolamento database con migliaia di articoli reali
Acquisizione da RSS feeds e NewsAPI
"""

import sys
import json
import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.settings import config_manager
from core.logger import get_logger, setup_logging
from ingestion.news_fetcher import NewsArticle
from processing.text_normalizer import normalize_text, assess_text_quality

setup_logging()
logger = get_logger(__name__)

# RSS Feeds affidabili
RSS_SOURCES = [
    {"url": "https://rss.cnn.com/rss/edition.rss", "source": "CNN"},
    {"url": "https://feeds.bbci.co.uk/news/rss.xml", "source": "BBC"},
    {"url": "http://rss.reuters.com/reuters/topNews", "source": "Reuters"},
    {"url": "https://www.theguardian.com/world/rss", "source": "Guardian"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "source": "NY Times"},
    {"url": "https://www.repubblica.it/rss/homepage/rss2.0.xml", "source": "La Repubblica"},
    {"url": "https://www.corriere.it/rss/homepage.xml", "source": "Corriere"},
    {"url": "https://www.ansa.it/sito/notizie/topnews/topnews_rss.xml", "source": "ANSA"},
    {"url": "https://feeds.skynews.com/feeds/rss/world.xml", "source": "Sky News"},
    {"url": "https://rss.dw.com/rdf/rss-en-all", "source": "Deutsche Welle"}
]

class RealDataIngestion:
    def __init__(self):
        self.articles_file = Path("data/real_articles.json")
        self.articles_file.parent.mkdir(exist_ok=True)
        self.articles = []
        self.processed_urls = set()
        
        # Carica articoli esistenti se presenti
        self.load_existing_articles()
    
    def load_existing_articles(self):
        """Carica articoli già salvati"""
        if self.articles_file.exists():
            try:
                with open(self.articles_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.articles = data.get('articles', [])
                    self.processed_urls = {art['url'] for art in self.articles}
                logger.info(f"Caricati {len(self.articles)} articoli esistenti")
            except Exception as e:
                logger.error(f"Errore caricamento articoli: {e}")
    
    def save_articles(self):
        """Salva articoli su file JSON"""
        try:
            data = {
                'last_update': datetime.now().isoformat(),
                'total_articles': len(self.articles),
                'articles': self.articles
            }
            with open(self.articles_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Salvati {len(self.articles)} articoli")
        except Exception as e:
            logger.error(f"Errore salvataggio: {e}")
    
    async def fetch_from_rss(self, session, source_info):
        """Fetch da singolo RSS feed"""
        try:
            url = source_info['url']
            source_name = source_info['source']
            
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return []
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                articles = []
                for entry in feed.entries[:20]:  # Prendi primi 20 per feed
                    if entry.link in self.processed_urls:
                        continue
                    
                    # Parsing data
                    try:
                        if hasattr(entry, 'published_parsed'):
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = datetime.now()
                    except:
                        pub_date = datetime.now()
                    
                    # Filtra articoli troppo vecchi (ultimi 7 giorni)
                    if pub_date < datetime.now() - timedelta(days=7):
                        continue
                    
                    # Crea oggetto articolo
                    article_data = {
                        'url': entry.link,
                        'title': entry.title,
                        'content': getattr(entry, 'description', ''),
                        'author': getattr(entry, 'author', 'Unknown'),
                        'published_date': pub_date.isoformat(),
                        'source': source_name,
                        'language': 'en' if source_name in ['CNN', 'BBC', 'Reuters', 'Guardian', 'NY Times'] else 'it'
                    }
                    
                    articles.append(article_data)
                
                logger.info(f"Estratti {len(articles)} articoli da {source_name}")
                return articles
                
        except Exception as e:
            logger.error(f"Errore RSS {source_info['source']}: {e}")
            return []
    
    async def fetch_all_rss(self):
        """Fetch da tutti i feed RSS"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_from_rss(session, source) for source in RSS_SOURCES]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            new_articles = []
            for result in results:
                if isinstance(result, list):
                    new_articles.extend(result)
            
            # Processo e filtra articoli
            for article_data in new_articles:
                if article_data['url'] not in self.processed_urls:
                    # Normalizza testo
                    normalized_content = normalize_text(article_data['content'])
                    quality = assess_text_quality(normalized_content)
                    
                    # Filtra per qualità
                    if quality > 0.4 and len(normalized_content) > 50:
                        article_data['content'] = normalized_content
                        article_data['quality_score'] = quality
                        self.articles.append(article_data)
                        self.processed_urls.add(article_data['url'])
            
            return len(new_articles)
    
    def get_articles_for_dashboard(self):
        """Converte articoli in formato per dashboard"""
        dashboard_articles = []
        
        for art_data in self.articles[-50]:  # Ultimi 50 per performance
            try:
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
                dashboard_articles.append(article)
            except Exception as e:
                logger.error(f"Errore conversione articolo: {e}")
                continue
        
        return dashboard_articles

async def populate_database():
    """Funzione principale per popolare il database"""
    print("=" * 60)
    print("VERIFACT - POPOLAZIONE DATABASE CON ARTICOLI REALI")
    print("=" * 60)
    
    ingestion = RealDataIngestion()
    
    print(f"Articoli esistenti: {len(ingestion.articles)}")
    print("Avvio acquisizione da RSS feeds...")
    
    try:
        new_count = await ingestion.fetch_all_rss()
        ingestion.save_articles()
        
        print(f"Nuovi articoli acquisiti: {new_count}")
        print(f"Totale articoli nel database: {len(ingestion.articles)}")
        
        # Statistiche
        sources_stats = {}
        for article in ingestion.articles:
            source = article['source']
            sources_stats[source] = sources_stats.get(source, 0) + 1
        
        print("\nDistribuzione per fonte:")
        for source, count in sources_stats.items():
            print(f"  {source}: {count} articoli")
        
        print(f"\nFile salvato in: {ingestion.articles_file}")
        print("Database popolato con successo!")
        
    except Exception as e:
        logger.error(f"Errore popolazione database: {e}")
        print(f"Errore: {e}")

def update_dashboard_articles():
    """Aggiorna dashboard con articoli reali"""
    ingestion = RealDataIngestion()
    articles = ingestion.get_articles_for_dashboard()
    
    print(f"Articoli disponibili per dashboard: {len(articles)}")
    
    # Crea file per dashboard
    dashboard_code = f'''
# Sostituisci la funzione get_demo_articles() con:

@st.cache_data
def get_real_articles():
    """Carica articoli reali dal database"""
    import json
    from pathlib import Path
    from datetime import datetime
    
    articles_file = Path("data/real_articles.json")
    if not articles_file.exists():
        return get_demo_articles()  # Fallback
    
    try:
        with open(articles_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        articles = []
        for art_data in data.get('articles', [])[-30]:  # Ultimi 30
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
        print(f"Errore caricamento articoli reali: {{e}}")
        return get_demo_articles()  # Fallback

# Poi sostituisci tutte le chiamate get_demo_articles() con get_real_articles()
'''
    
    with open("dashboard_update.txt", "w", encoding="utf-8") as f:
        f.write(dashboard_code)
    
    print("Istruzioni salvate in dashboard_update.txt")
    print("Copia il codice per aggiornare il dashboard")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Popola database con articoli reali')
    parser.add_argument('--action', choices=['populate', 'update-dashboard'], 
                       default='populate', help='Azione da eseguire')
    
    args = parser.parse_args()
    
    if args.action == 'populate':
        asyncio.run(populate_database())
    elif args.action == 'update-dashboard':
        update_dashboard_articles()