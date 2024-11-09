from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from typing import List
from sqlalchemy import select
from schemas import ModelResponseSchema, DetailSchema
from models import Detail
from database import db_dependency
import pandas as pd

app = FastAPI()

@app.post('/get_detail')
async def get_detail(
    db: db_dependency,
    model_response: ModelResponseSchema = Body(...)
) -> DetailSchema | None:
    detail_orm: Detail | None = \
        (await db
         .execute(
            select(Detail)
            .filter(
                Detail.detail_article == model_response.detail_article and
                Detail.detail_number == model_response.detail_number
            )
        )) \
        .scalars() \
        .one_or_none()

    return DetailSchema.model_validate(detail_orm) if detail_orm is not None else None

@app.post('/add_details')
async def add_details(
    db: db_dependency,
    xlsx_file: UploadFile = File(...)
) -> List[DetailSchema]:
    xlsx_content: bytes = await xlsx_file.read()
    df: pd.DataFrame = pd.read_excel(xlsx_content)

    detail_orm_list: List[Detail] = [Detail(*row) for row in df.itertuples(index=False)]

    db.add_all(detail_orm_list)
    await db.commit()

    return [DetailSchema.model_validate(detail_orm) for detail_orm in detail_orm_list]