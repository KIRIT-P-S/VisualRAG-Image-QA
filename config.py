import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.5-flash" # Free tier model

    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"

    PDF_DPI: int = 300          
    MAX_IMAGE_SIZE: tuple = (1024, 1024)  

    EMBEDDING_DIM: int = 512    
    FAISS_INDEX_TYPE: str = "flat"   

    DEFAULT_TOP_K: int = 3
    MAX_TOKENS_GENERATION: int = 2048

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
