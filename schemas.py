from pydantic import BaseModel
from typing import List

class ModelResponseSchema(BaseModel):
    detail_article: str
    detail_number: int

class DetailSchema(BaseModel):
    detail_article: str
    detail_number: int
    detail_name: str
    order_number: int
    statio_block: str

    class Config:
        from_attributes: bool = True