from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy.orm import Mapped, mapped_column, relationship

from .types import bigint, bigint_u, str_512, Base

# Circular import fix
if TYPE_CHECKING:  # pragma: no cover
    from .chat_user import ChatUser


class User(Base):
    __tablename__ = "users"

    # The internal identifier of each user
    id: Mapped[bigint_u] = mapped_column(primary_key=True, autoincrement=True)

    # ---------------------------------------------------------------------------- #
    #                              Telegram identifier                             #
    # ---------------------------------------------------------------------------- #
    telegram_id: Mapped[bigint] = mapped_column(unique=True, index=True)
    name: Mapped[Optional[str_512]]  # At time of crawling, this is the username

    # ---------------------------------------------------------------------------- #
    #                               Telegram metadata                              #
    # ---------------------------------------------------------------------------- #
    first_name: Mapped[Optional[str_512]]
    last_name: Mapped[Optional[str_512]]

    language_code: Mapped[Optional[str_512]]
    phone_number: Mapped[Optional[int]]
    phone_number_country_code: Mapped[Optional[int]]

    is_verified: Mapped[bool] = mapped_column(server_default="0")
    is_restricted: Mapped[bool] = mapped_column(server_default="0")
    is_bot: Mapped[bool] = mapped_column(server_default="0")
    is_fake: Mapped[bool] = mapped_column(server_default="0")
    is_scam: Mapped[bool] = mapped_column(server_default="0")
    is_support: Mapped[bool] = mapped_column(server_default="0")
    is_deleted: Mapped[bool] = mapped_column(server_default="0")
    is_premium: Mapped[bool] = mapped_column(server_default="0")

    # ---------------------------------------------------------------------------- #
    #                         Relationships / Foreign Keys                         #
    # ---------------------------------------------------------------------------- #
    # many-to-many relationship to Childs
    chatUsers: Mapped[Optional[List[ChatUser]]] = relationship(
        back_populates="user",
        enable_typechecks=False,
    )

    def __repr__(self):  # pragma: no cover
        short_name = str(self.name)
        if len(short_name) > 10:
            short_name = short_name[0:10]
        return f"<User '@{short_name}'>"
