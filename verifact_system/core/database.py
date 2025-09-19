"""
VeriFact System - PostgreSQL Database Manager
Database manager aggiornato per PostgreSQL
"""

import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from core.settings import DATABASE_CONFIG
from core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Article:
    """Struttura dati articolo per database"""
    id: Optional[int]
    url: str
    title: str
    content: str
    author: str
    published_date: datetime
    source: str
    language: str
    quality_score: float
    created_at: Optional[datetime] = None

class PostgreSQLManager:
    """Manager PostgreSQL per VeriFact"""
    
    def __init__(self):
        self.config = DATABASE_CONFIG
        self.connection_pool = None
        self.database_url = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
    
    async def initialize_pool(self):
        """Inizializza pool di connessioni"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=10,
                command_timeout=60
            )
            logger.info("Pool connessioni PostgreSQL inizializzato")
        except Exception as e:
            logger.error(f"Errore inizializzazione pool: {e}")
            raise
    
    async def get_articles(self, limit: int = 100, offset: int = 0, source: str = None) -> List[Article]:
        """Recupera articoli dal database"""
        if not self.connection_pool:
            await self.initialize_pool()
        
        query = "SELECT * FROM articles"
        params = []
        
        if source:
            query += " WHERE source = $1"
            params.append(source)
        
        query += " ORDER BY published_date DESC LIMIT $%d OFFSET $%d" % (len(params)+1, len(params)+2)
        params.extend([limit, offset])
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            articles = []
            for row in rows:
                article = Article(
                    id=row['id'],
                    url=row['url'],
                    title=row['title'],
                    content=row['content'],
                    author=row['author'],
                    published_date=row['published_date'],
                    source=row['source'],
                    language=row['language'],
                    quality_score=row['quality_score'],
                    created_at=row['created_at']
                )
                articles.append(article)
            
            return articles
    
    async def insert_article(self, article_data: Dict[str, Any]) -> int:
        """Inserisce nuovo articolo"""
        if not self.connection_pool:
            await self.initialize_pool()
        
        async with self.connection_pool.acquire() as conn:
            article_id = await conn.fetchval("""
                INSERT INTO articles (url, title, content, author, published_date, source, language, quality_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (url) DO NOTHING
                RETURNING id
            """,
                article_data['url'],
                article_data['title'],
                article_data['content'],
                article_data.get('author', 'Unknown'),
                article_data['published_date'],
                article_data['source'],
                article_data.get('language', 'en'),
                article_data.get('quality_score', 0.5)
            )
            
            return article_id
    
    async def search_articles(self, keywords: List[str], limit: int = 20) -> List[Article]:
        """Ricerca articoli per keywords"""
        if not self.connection_pool:
            await self.initialize_pool()
        
        # Crea query di ricerca fulltext
        search_query = " | ".join(keywords)  # OR tra keywords
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT *, 
                       ts_rank(to_tsvector('english', title || ' ' || content), to_tsquery('english', $1)) as rank
                FROM articles 
                WHERE to_tsvector('english', title || ' ' || content) @@ to_tsquery('english', $1)
                ORDER BY rank DESC
                LIMIT $2
            """, search_query, limit)
            
            articles = []
            for row in rows:
                article = Article(
                    id=row['id'],
                    url=row['url'],
                    title=row['title'],
                    content=row['content'],
                    author=row['author'],
                    published_date=row['published_date'],
                    source=row['source'],
                    language=row['language'],
                    quality_score=row['quality_score'],
                    created_at=row['created_at']
                )
                articles.append(article)
            
            return articles
    
    async def get_stats(self) -> Dict[str, Any]:
        """Statistiche database"""
        if not self.connection_pool:
            await self.initialize_pool()
        
        async with self.connection_pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM articles")
            
            sources = await conn.fetch("""
                SELECT source, COUNT(*) as count 
                FROM articles 
                GROUP BY source 
                ORDER BY count DESC
            """)
            
            recent = await conn.fetchval("""
                SELECT COUNT(*) FROM articles 
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            
            avg_quality = await conn.fetchval("""
                SELECT AVG(quality_score) FROM articles
            """)
            
            return {
                'total_articles': total,
                'sources_distribution': {row['source']: row['count'] for row in sources},
                'articles_last_24h': recent,
                'average_quality': float(avg_quality) if avg_quality else 0.0
            }
    
    async def close(self):
        """Chiude pool connessioni"""
        if self.connection_pool:
            await self.connection_pool.close()

# Istanza globale
db_manager = PostgreSQLManager()

# Funzioni di utilità
async def get_articles_from_db(limit: int = 50) -> List[Article]:
    """Utility per recuperare articoli"""
    return await db_manager.get_articles(limit=limit)

async def search_articles_by_keywords(keywords: List[str]) -> List[Article]:
    """Utility per ricerca"""
    return await db_manager.search_articles(keywords)

async def get_database_stats() -> Dict[str, Any]:
    """Utility per statistiche"""
    return await db_manager.get_stats()