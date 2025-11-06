from sqlalchemy import ForeignKey, String, Index
from sqlalchemy.orm import Mapped, mapped_column

from telegram_data_models import Base


class ChatLanguage(Base):
    __tablename__ = 'chat_language'
    __bind_key__ = "Base"

    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"), primary_key=True)
    lang: Mapped[str] = mapped_column(String(8), nullable=False)
    score: Mapped[float] = mapped_column(nullable=False)
    columns_used: Mapped[str] = mapped_column(nullable=False)
    
    __table_args__ = (
        Index('idx_chatlang_lang_score', 'lang', 'score'),
    )