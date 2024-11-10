from fastapi import FastAPI, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from sqlalchemy import select
from schemas import ModelResponseSchema, DetailSchema
from models import Detail
from database import db_dependency
import pandas as pd
from PIL import Image
from io import BytesIO

from ml.image_processing import predict_on_image, get_detail_dataset_info
from ml.init_models import get_text_recognizer, get_text_box_detector

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/model')
async def image_to_text(
    file: UploadFile = File(...)
) -> ModelResponseSchema:
    """
    Данный эндпоинт связывает сервер с API модели, на входе подается файл,
    на выходе получаем артикул и номер детали и баундинг бокс
    :param file
    :return: model_response
    """

    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes))

    text, bbox = predict_on_image(image, *get_detail_dataset_info(), *get_text_recognizer(), get_text_box_detector())
    article, number = text[1:-1].rsplit(' ', 1)

    return ModelResponseSchema(
        detail_article=article,
        detail_number=number
    )

@app.post('/get_detail')
async def get_detail(
    db: db_dependency,
    model_response: ModelResponseSchema = Body(...)
) -> DetailSchema | None:
    """
    Данный эндпоинт по номеру и артикулу позволяет получить информацию о детали с базы
    :param db, model_response = {article, number}
    :return: detail_schema
    """

    detail_orm: Detail | None = \
        (await db
         .execute(
            select(Detail)
            .where(
                Detail.detail_article.contains(model_response.detail_article),
                Detail.detail_number == model_response.detail_number
            )
        )) \
        .scalars() \
        .one_or_none()

    return DetailSchema.model_validate(detail_orm) if detail_orm else None

@app.post('/add_details')
async def add_details(
    db: db_dependency,
    xlsx_file: UploadFile = File(...)
) -> List[DetailSchema]:
    """
    Данный эндпоинт отвечает за загрузку данных из xsl в базу данных
    :param db, xlsx_file
    :return: detail_schema
    """

    xlsx_content: BytesIO = await xlsx_file.read()
    df: pd.DataFrame = pd.read_excel(xlsx_content, header=0)

    detail_orm_list: List[Detail] = [
        Detail(*row.values)
        for _, row in df.iterrows()
    ]

    db.add_all(detail_orm_list)
    await db.commit()

    return [DetailSchema.model_validate(detail_orm) for detail_orm in detail_orm_list]