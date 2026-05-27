from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Circular import fix
if TYPE_CHECKING:  # pragma: no cover
    from .chat_user import ChatUser
    from .message import Message

from .types import bigint, bigint_u, str_512, str_8, text, Base


class Chat(Base):
    """A model to represent a chat or channel on Telegram."""

    __tablename__ = "chats"
    __bind_key__ = "Base"

    # The internal identifier of each chat
    id: Mapped[bigint_u] = mapped_column(primary_key=True, autoincrement=True)

    # ---------------------------------------------------------------------------- #
    #                              Telegram identifier                             #
    # ---------------------------------------------------------------------------- #
    telegram_id: Mapped[bigint] = mapped_column(unique=True, index=True)
    name: Mapped[Optional[str_512]]  # At time of crawling, this is the username

    # ---------------------------------------------------------------------------- #
    #                               Telegram metadata                              #
    # ---------------------------------------------------------------------------- #
    # Title, for supergroups, channels and basic group chats.
    title: Mapped[Optional[str_512]]
    # Type of the chat possible values are BOT, PRIVATE, GROUP, SUPERGROUP, CHANNEL
    type: Mapped[Optional[str_512]]
    # Sometimes a description is added to the chat
    description: Mapped[Optional[text]]
    # Not every chat has a members count set visible
    members_count: Mapped[Optional[int]]

    # This can be get from the telegram api
    # but no idea how it is set or how
    # the flags are created
    is_verified: Mapped[bool] = mapped_column(server_default="0")
    is_restricted: Mapped[bool] = mapped_column(server_default="0")
    is_fake: Mapped[bool] = mapped_column(server_default="0")
    is_scam: Mapped[bool] = mapped_column(server_default="0")
    is_support: Mapped[bool] = mapped_column(server_default="0")

    # Autodelete timer (if active in chats)
    message_auto_delete_time: Mapped[Optional[int]]

    # Crawling status
    status: Mapped[str]

    # ---------------------------------------------------------------------------- #
    #                         Relationships / Foreign Keys                         #
    # ---------------------------------------------------------------------------- #

    # It is possible for a linked chat to be present i.e. a discussion group
    # This channel is a own chat row in the database
    linked_chat_id: Mapped[Optional[bigint_u]] = mapped_column(ForeignKey("chats.id"), index=True)
    linked_chat: Mapped[Optional[Chat]] = relationship(
        remote_side=[id],
        foreign_keys=[linked_chat_id],
        post_update=True,
        enable_typechecks=False,
    )

    # many-to-many relationship to Childs
    messages: Mapped[Optional[List[Message]]] = relationship(
        back_populates="chat", foreign_keys="Message.chat_id", enable_typechecks=False
    )
    users: Mapped[Optional[List[ChatUser]]] = relationship(
        back_populates="chat", enable_typechecks=False
    )


class ChatLanguage(Base):
    __tablename__ = 'chat_language'
    __bind_key__ = "Base"

    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"), primary_key=True)
    lang: Mapped[str_8] = mapped_column(nullable=False)
    score: Mapped[float] = mapped_column(nullable=False)
    columns_used: Mapped[str] = mapped_column(nullable=False)

    chat: Mapped[Optional[Chat]] = relationship(
        remote_side=[Chat.id],
        foreign_keys=[chat_id],
        enable_typechecks=False,
    )

    __table_args__ = (Index('idx_chatlang_lang_score', 'lang', 'score'),)
