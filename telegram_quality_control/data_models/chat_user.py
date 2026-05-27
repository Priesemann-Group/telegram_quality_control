from __future__ import annotations

from typing import Optional

from sqlalchemy import FetchedValue, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .types import bigint_u, str_256, timestamp, Base
from .chat import Chat
from .user import User


class ChatUser(Base):
    """A model to represent the many-to-many relationship between chats and users."""

    __tablename__ = "chats_users"

    # The internal identifier of each chat
    chat_id: Mapped[bigint_u] = mapped_column(
        ForeignKey("chats.id"), primary_key=True, server_default=FetchedValue()
    )
    user_id: Mapped[bigint_u] = mapped_column(
        ForeignKey("users.id"), primary_key=True, server_default=FetchedValue()
    )

    # Extra information about the user in the chat
    joined: Mapped[Optional[timestamp]] = mapped_column(nullable=True)
    status: Mapped[Optional[str_256]] = mapped_column(nullable=True)

    # ---------------------------------------------------------------------------- #
    #                              association proxies                             #
    # ---------------------------------------------------------------------------- #
    chat: Mapped[Optional[Chat]] = relationship(
        remote_side=[Chat.id],
        foreign_keys=[chat_id],
        enable_typechecks=False,
    )

    user: Mapped[Optional[User]] = relationship(
        remote_side=[User.id],
        foreign_keys=[user_id],
        enable_typechecks=False,
    )

    def __repr__(self):  # pragma: no cover
        """Return a string representation of the ChatUser model."""
        return f"<ChatUser '@{self.chat_id}&{self.user_id}'>"
