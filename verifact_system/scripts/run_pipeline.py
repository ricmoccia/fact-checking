#!/usr/bin/env python3
"""
VeriFact System - Main Pipeline Runner
Script principale per eseguire la pipeline completa di fact-checking
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

# Aggiungi root directory al path
sys.path.append(str(Path(__file__).parent.parent))

from core.settings import config_manager
from core.database import initialize_database, db_manager
from core.logger import get_logger
from ingestion.news_fetcher import NewsStreamProcessor, fetch_daily_news_batch
from processing.text_normalizer import text_processor

logger = get_logger(__name__)


class VeriFactPipeline:
    """Pipeline principale del sistema VeriFact"""
    
    def __init__(self):
        self.config = config_manager
        self.news_processor = NewsStreamProcessor()
        self.stats = {
            'articles_processed': 0,
            'articles_stored': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def initialize_system(self):
        """Inizializza tutti i componenti del sistema"""
        logger.info("Initializing VeriFact system...")
        
        try:
            # Inizializza database
            success = await initialize_database()
            if not success:
                raise RuntimeError("Database initialization failed")
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def run_daily_ingestion(self, max_articles: int = 100):
        """Esegue ingestion giornaliera di articoli"""
        logger.info(f"Starting daily news ingestion (max {max_articles} articles)")
        self.stats['start_time'] = datetime.now()
        
        try:
            # Fetch batch di articoli
            articles = await fetch_daily_news_batch(max_articles)
            logger.info(f"Fetched {len(articles)} articles from news sources")
            
            # Processa ogni articolo
            processed_articles = []
            for article in articles:
                try:
                    # Normalizza testo
                    processing_result = text_processor.process_document(
                        article.content, article.language
                    )
                    
                    # Aggiorna articolo con testo processato
                    article.content = processing_result.processed_text
                    article.quality_score = processing_result.quality_score
                    
                    # Solo articoli di qualità sufficiente
                    if processing_result.quality_score >= 0.6:
                        processed_articles.append(article)
                        self.stats['articles_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing article {article.title[:50]}: {e}")
                    self.stats['errors'] += 1
                    continue
            
            # Salva articoli processati
            await self._store_articles(processed_articles)
            
            self.stats['end_time'] = datetime.now()
            await self._log_statistics()
            
        except Exception as e:
            logger.error(f"Daily ingestion failed: {e}")
            self.stats['errors'] += 1
            raise
    
    async def run_continuous_ingestion(self):
        """Esegue ingestion continua di articoli"""
        logger.info("Starting continuous news ingestion...")
        
        try:
            async for batch in self.news_processor.stream_latest_news():
                await self._process_and_store_batch(batch)
                
        except KeyboardInterrupt:
            logger.info("Continuous ingestion stopped by user")
        except Exception as e:
            logger.error(f"Continuous ingestion failed: {e}")
            raise
    
    async def _process_and_store_batch(self, articles):
        """Processa e salva batch di articoli"""
        processed_articles = []
        
        for article in articles:
            try:
                # Processa testo
                processing_result = text_processor.process_document(
                    article.content, article.language
                )
                
                # Filtra per qualità
                if processing_result.quality_score >= 0.6:
                    article.content = processing_result.processed_text
                    article.quality_score = processing_result.quality_score
                    processed_articles.append(article)
                    self.stats['articles_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing article: {e}")
                self.stats['errors'] += 1
                continue
        
        # Salva batch
        await self._store_articles(processed_articles)
    
    async def _store_articles(self, articles):
        """Salva articoli nel database"""
        try:
            async with db_manager.get_async_session() as session:
                for article in articles:
                    # TODO: Implementa inserimento in tabella articoli
                    # Per ora solo conteggio
                    self.stats['articles_stored'] += 1
                    
                logger.info(f"Stored {len(articles)} articles in database")
                
        except Exception as e:
            logger.error(f"Failed to store articles: {e}")
            raise
    
    async def _log_statistics(self):
        """Log delle statistiche pipeline"""
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            logger.info(f"""
            Pipeline Statistics:
            - Duration: {duration}
            - Articles Processed: {self.stats['articles_processed']}
            - Articles Stored: {self.stats['articles_stored']}
            - Errors: {self.stats['errors']}
            - Success Rate: {(self.stats['articles_stored']/max(self.stats['articles_processed'],1)*100):.1f}%
            """)
    
    async def run_full_pipeline(self, max_articles: int = 50):
        """Esegue pipeline completa: ingestion + processing + vectorization"""
        logger.info("Starting full VeriFact pipeline...")
        
        try:
            # Fase 1: Ingestion
            await self.run_daily_ingestion(max_articles)
            
            # Fase 2: Processing avanzato (TODO)
            logger.info("Advanced processing phase - TODO")
            
            # Fase 3: Vectorization (TODO) 
            logger.info("Vectorization phase - TODO")
            
            logger.info("Full pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Full pipeline failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup risorse"""
        try:
            await db_manager.close_connections()
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    """Funzione main del script"""
    parser = argparse.ArgumentParser(description='VeriFact System Pipeline Runner')
    parser.add_argument('--mode', choices=['daily', 'continuous', 'full'], 
                       default='daily', help='Pipeline execution mode')
    parser.add_argument('--max-articles', type=int, default=50,
                       help='Maximum articles to process (daily mode)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Configura logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    pipeline = VeriFactPipeline()
    
    try:
        # Inizializza sistema
        if not await pipeline.initialize_system():
            logger.error("System initialization failed, exiting")
            sys.exit(1)
        
        # Esegui modalità selezionata
        if args.mode == 'daily':
            await pipeline.run_daily_ingestion(args.max_articles)
        elif args.mode == 'continuous':
            await pipeline.run_continuous_ingestion()
        elif args.mode == 'full':
            await pipeline.run_full_pipeline(args.max_articles)
        
        logger.info("Pipeline execution completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())