from fastapi import FastAPI, Body
from schemas import ModelResponseSchema, DetailSchema
from models import Detail
from database import db_dependency

app = FastAPI()

@app.post('/add_detail')
async def add_detail(
        db: db_dependency,
        model_response: ModelResponseSchema = Body(...),
        detail_name: str = Body(...),
        order_number: int = Body(...),
        station_block: str = Body(...)
) -> DetailSchema:
    detail_orm: Detail = Detail(
        detail_article=model_response.detail_article,
        detail_number=model_response.detail_number,
        detail_name=detail_name,
        order_number=order_number,
        station_block=station_block
    )

    db.add(detail_orm)
    await db.commit()
    await db.refresh(detail_orm)

    return DetailSchema.model_validate(detail_orm)
