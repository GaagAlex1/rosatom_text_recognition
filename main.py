from fastapi import FastAPI, Body
from typing import List
from schemas import ModelResponseSchema, DetailSchema
from models import Detail
from database import db_dependency

app = FastAPI()

@app.post('/add_detail')
async def add_detail(
    db: db_dependency,
    model_response: ModelResponseSchema = Body(...)
) -> DetailSchema:
    detail_orm: Detail = Detail(
        detail_article=model_response.detail_article,
        detail_number=model_response.detail_number
    )

    db.add(detail_orm)
    await db.commit()
    await db.refresh(detail_orm)

    return DetailSchema.model_validate(detail_orm)

@app.post('/add_details')
async def add_details(
    db: db_dependency,
    model_response_list: List[ModelResponseSchema] = Body(...)
) -> List[DetailSchema]:
    detail_orm_list: List[Detail] = [
        Detail(
            detail_article=model_response.detail_article,
            detail_number=model_response.detail_number
        )
        for model_response in model_response_list
    ]

    db.add_all(detail_orm_list)
    await db.commit()

    return [DetailSchema.model_validate(detail_orm) for detail_orm in detail_orm_list]