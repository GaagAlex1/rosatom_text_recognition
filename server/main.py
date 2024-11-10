from fastapi import FastAPI, Body, UploadFile, File
from typing import List
from sqlalchemy import select
from schemas import ModelResponseSchema, DetailSchema
from models import Detail
from database import db_dependency
import pandas as pd
from PIL import Image
from io import BytesIO
from ml.image_processing import get_text_and_box_from_image
from ml.init_models import get_text_recognizer, get_text_box_detector

app = FastAPI()

@app.post('/model')
async def image_to_text(
    file: UploadFile = File(...)
) -> ModelResponseSchema:
    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes))

    text, bbox = get_text_and_box_from_image(image, *get_text_recognizer(), get_text_box_detector())
    print(text, bbox)
    return ModelResponseSchema(
        detail_article=text,
        detail_number=0
    )

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
    xlsx_content: BytesIO = await xlsx_file.read()
    df: pd.DataFrame = pd.read_excel(xlsx_content, header=0)

    detail_orm_list: List[Detail] = [
        Detail(*row.values)
        for _, row in df.iterrows()
    ]

    db.add_all(detail_orm_list)
    await db.commit()

    return [DetailSchema.model_validate(detail_orm) for detail_orm in detail_orm_list]