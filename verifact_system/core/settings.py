import os
import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "verifact"
    username: str = "postgres" 
    password: str = "verifact123"
    
@dataclass
class APIConfig:
    news_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

@dataclass
class ModelConfig:
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "gemma:3b"
    embedding_dimension: int = 384
    max_chunk_size: int = 512
    chunk_overlap: int = 50

class ConfigManager:
    def __init__(self):
        self.database_config = DatabaseConfig()
        self.api_config = APIConfig()
        self.model_config = ModelConfig()
        
        # Carica da environment variables se presenti
        self._load_from_env()
    
    def _load_from_env(self):
        # Database
        if os.getenv('DATABASE_HOST'):
            self.database_config.host = os.getenv('DATABASE_HOST')
        if os.getenv('DATABASE_PORT'):
            self.database_config.port = int(os.getenv('DATABASE_PORT'))
        if os.getenv('DATABASE_NAME'):
            self.database_config.database = os.getenv('DATABASE_NAME')
        if os.getenv('DATABASE_USER'):
            self.database_config.username = os.getenv('DATABASE_USER')
        if os.getenv('DATABASE_PASSWORD'):
            self.database_config.password = os.getenv('DATABASE_PASSWORD')
            
        # API Keys
        if os.getenv('NEWS_API_KEY'):
            self.api_config.news_api_key = os.getenv('NEWS_API_KEY')
        if os.getenv('OPENAI_API_KEY'):
            self.api_config.openai_api_key = os.getenv('OPENAI_API_KEY')
    
    @property
    def database(self):
        return self.database_config
    
    @property
    def apis(self):
        return self.api_config
    
    @property
    def models(self):
        return self.model_config
    
    def get_news_sources(self):
        return [
            {
                'name': 'NewsAPI',
                'url': 'https://newsapi.org/v2/top-headlines',
                'enabled': True,
                'quality_weight': 0.8
            }
        ]
    
    def get_processing_config(self):
        return {
            'batch_size': 16,
            'max_workers': 2,
            'text_quality_threshold': 0.6,
            'supported_languages': ['en', 'it', 'es', 'fr', 'de']
        }

# Istanza globale
config_manager = ConfigManager()

# Export per compatibilità
DATABASE_CONFIG = config_manager.database
API_CONFIG = config_manager.apis
MODEL_CONFIG = config_manager.models