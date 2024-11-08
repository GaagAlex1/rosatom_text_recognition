from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

class Settings(BaseSettings):
    db_host: str
    db_port: int
    db_user: str
    db_pass: str
    db_name: str

    @property
    def DATABASE_URL_asyncpg(self):
        return URL.create('postgresql+asyncpg',
                          username=self.db_user,
                          password=self.db_pass,
                          host=self.db_host,
                          port=self.db_port,
                          database=self.db_name)

    model_config = SettingsConfigDict(env_file='.env', case_sensitive=False)


settings = Settings()