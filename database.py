from typing import AsyncGenerator, Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from config import settings
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


url = settings.db.DATABASE_URL_asyncpg
async_engine = create_async_engine(url)
async_session_maker = async_sessionmaker(async_engine, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session

db_dependency = Annotated[AsyncSession, Depends(get_async_session)]