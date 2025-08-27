from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "News Intelligence API"
    db_url: str = "postgresql+psycopg://news:newspass@localhost:5432/newsdb"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
