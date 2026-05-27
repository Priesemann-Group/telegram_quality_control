from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .types import bigint_u, longtext, str_256, timestamp, Base
from .types import text as SQLText

if TYPE_CHECKING:  # pragma: no cover
    from .message import Message


class Poll(Base):
    __tablename__ = "polls"

    id: Mapped[bigint_u] = mapped_column(primary_key=True, autoincrement=True)

    # Mapping to the message (one-to-one relationship)
    message_id: Mapped[bigint_u] = mapped_column(ForeignKey("messages.id"), unique=True)
    message: Mapped[Message] = relationship(
        "Message",
        back_populates="poll",
        enable_typechecks=False,
    )

    # The poll type
    type: Mapped[str_256]

    # The poll question
    question: Mapped[str_256]
    explanation: Mapped[Optional[str_256]]

    # Some metadata about the poll sometimes not available
    total_voters: Mapped[Optional[int]]
    is_anonymous: Mapped[Optional[bool]]
    allows_multiple_answers: Mapped[Optional[bool]]
    is_closed: Mapped[Optional[bool]]
    close_date: Mapped[Optional[timestamp]]

    # The poll options
    options: Mapped[list[PollOption]] = relationship(
        "PollOption",
        back_populates="poll",
        enable_typechecks=False,
    )

    def __repr__(self):  # pragma: no cover
        return f"<Poll '@{self.id}'>"


class PollOption(Base):
    __tablename__ = "poll_options"

    # The internal identifier of each poll option
    id: Mapped[bigint_u] = mapped_column(primary_key=True, autoincrement=True)

    # Mapping to the messages (one-to-many relationship)
    poll_id: Mapped[bigint_u] = mapped_column(ForeignKey("polls.id"))
    poll: Mapped[Message] = relationship(
        "Poll",
        remote_side=[Poll.id],
        foreign_keys=[poll_id],
        enable_typechecks=False,
    )

    text: Mapped[SQLText]
    voters: Mapped[int]

    data: Mapped[Optional[longtext]]

    def __repr__(self):  # pragma: no cover
        return f"<PollOption '@{self.id}'>"
