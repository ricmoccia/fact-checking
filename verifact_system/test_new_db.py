# Crea test_new_db.py
import asyncio
from core.database import db_manager

async def test_db():
    articles = await db_manager.get_articles(limit=5)
    print(f"Articoli trovati: {len(articles)}")
    
    stats = await db_manager.get_stats()
    print(f"Statistiche: {stats}")
    
    await db_manager.close()

asyncio.run(test_db())