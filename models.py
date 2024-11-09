from database import Base
from sqlalchemy.orm import mapped_column, Mapped

class Detail(Base):
    __tablename__ = 'detail'

    detail_article: Mapped[str] = mapped_column(primary_key=True)
    detail_number: Mapped[int] = mapped_column(primary_key=True)
    detail_name: Mapped[str] = mapped_column()
    order_number: Mapped[int] = mapped_column()
    station_block: Mapped[str] = mapped_column()
