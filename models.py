from database import Base
from sqlalchemy.orm import mapped_column, Mapped

class Detail(Base):
    __tablename__ = 'detail'

    detail_article: Mapped[str] = mapped_column(primary_key=True)
    detail_number: Mapped[int] = mapped_column(primary_key=True)
    detail_name: Mapped[str] = mapped_column()
    order_number: Mapped[int] = mapped_column()
    station_block: Mapped[str] = mapped_column()

    def __init__(
        self,
        detail_article: str,
        detail_number: int,
        detail_name: str,
        order_number: int,
        station_block: str
    ) -> None:
        super().__init__()
        self.detail_article = detail_article
        self.detail_number = detail_number
        self.detail_name = detail_name
        self.order_number = order_number
        self.station_block = station_block




