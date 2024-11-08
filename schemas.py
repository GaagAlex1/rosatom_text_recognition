from pydantic import BaseModel
from typing import List

class ModelResponseSchema(BaseModel):
    detail_article: str
    detail_number: int

class ModelResponsesSchema(BaseModel):
    detail_articles: List[str]
    detail_numbers: List[str]

class DetailSchema(BaseModel):
    id: int
    detail_article: str
    detail_number: int

    class Config:
        from_attributes: bool = True