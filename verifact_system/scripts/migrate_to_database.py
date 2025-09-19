"""
Migrazione articoli da JSON a PostgreSQL
"""
import json
import asyncio
import asyncpg
from datetime import datetime
from pathlib import Path

DATABASE_URL = "postgresql://postgres:verifact123@localhost:5432/verifact"

async def create_tables():
    """Crea tabelle nel database"""
    conn = await asyncpg.connect(DATABASE_URL)
    
    await conn.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            url VARCHAR(512) UNIQUE,
            title TEXT,
            content TEXT,
            author VARCHAR(256),
            published_date TIMESTAMP,
            source VARCHAR(256),
            language VARCHAR(10),
            quality_score FLOAT DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
        CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(published_date);
        CREATE INDEX IF NOT EXISTS idx_articles_language ON articles(language);
    """)
    
    await conn.close()
    print("Tabelle create!")

async def migrate_articles():
    """Migra articoli da JSON a database"""
    articles_file = Path("data/real_articles.json")
    
    if not articles_file.exists():
        print("File JSON non trovato!")
        return
    
    with open(articles_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conn = await asyncpg.connect(DATABASE_URL)
    
    migrated = 0
    skipped = 0
    
    for article_data in data.get('articles', []):
        try:
            await conn.execute("""
                INSERT INTO articles (url, title, content, author, published_date, source, language, quality_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (url) DO NOTHING
            """, 
                article_data['url'],
                article_data['title'],
                article_data['content'],
                article_data.get('author', 'Unknown'),
                datetime.fromisoformat(article_data['published_date']),
                article_data['source'],
                article_data.get('language', 'en'),
                article_data.get('quality_score', 0.5)
            )
            migrated += 1
        except Exception as e:
            print(f"Errore migrazione articolo: {e}")
            skipped += 1
    
    await conn.close()
    print(f"Migrazione completata: {migrated} articoli migrati, {skipped} saltati")

async def verify_migration():
    """Verifica migrazione"""
    conn = await asyncpg.connect(DATABASE_URL)
    
    count = await conn.fetchval("SELECT COUNT(*) FROM articles")
    sources = await conn.fetch("SELECT source, COUNT(*) as count FROM articles GROUP BY source ORDER BY count DESC")
    
    print(f"\nVerifica migrazione:")
    print(f"Totale articoli: {count}")
    print("Distribuzione fonti:")
    for row in sources:
        print(f"  {row['source']}: {row['count']} articoli")
    
    await conn.close()

async def main():
    print("Inizio migrazione a PostgreSQL...")
    await create_tables()
    await migrate_articles()
    await verify_migration()

if __name__ == "__main__":
    asyncio.run(main())