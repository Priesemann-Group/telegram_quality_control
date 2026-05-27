from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .types import bigint_u, str_256, str_1024, str_4096, timestamp, Base
from .chat import Chat
from .poll import Poll
from .user import User


class Message(Base):
    __tablename__ = "messages"

    # The internal identifier of each chat
    id: Mapped[bigint_u] = mapped_column(primary_key=True, autoincrement=True)

    # The corresponding chat where the message was sent
    chat_id: Mapped[Optional[bigint_u]] = mapped_column(ForeignKey("chats.id"))

    """ Sender of the message
        ----------------------
    if sent on behalf of a chat.
    The channel itself for channel messages.
    The supergroup itself for messages from anonymous group administrators.
    The linked channel for messages automatically forwarded to the discussion group.
    """
    # User who sent the message (empty if it's a channel)
    from_user_id: Mapped[Optional[bigint_u]] = mapped_column(ForeignKey("users.id"))
    sender_chat_id: Mapped[Optional[bigint_u]] = mapped_column(ForeignKey("chats.id"))
    sender_chat = relationship("Chat", remote_side=[Chat.id], foreign_keys=[sender_chat_id])

    """ Forwards
        --------
        All optional fields are only available if the message is forwarded.
    """
    # For forwarded messages, sender of the original message.
    forward_from_id: Mapped[Optional[bigint_u]] = mapped_column(ForeignKey("users.id"))
    # For messages forwarded from channels, the original chat
    forward_from_chat_id: Mapped[Optional[bigint_u]] = mapped_column(ForeignKey("chats.id"))
    # The date the message was originally sent
    forward_date: Mapped[Optional[timestamp]]

    """ Replies
        -------
        All optional fields are only available if the message is a reply.
    """
    # The id of the message which this message is a reply to
    reply_to_message_id: Mapped[Optional[bigint_u]] = mapped_column(ForeignKey("messages.id"))

    # The id of the root message of the reply tree (in the case of comments to
    # channel posts, this will be the original post from the channel)
    reply_to_top_message_id: Mapped[Optional[bigint_u]] = mapped_column(ForeignKey("messages.id"))

    """ Content
        -------

        All as oneway link from the content to the message. They do not
        need an explicit reference here.
    """
    text_content: Mapped[Optional[MessageTextContent]] = relationship(
        back_populates="message",
        enable_typechecks=False,
    )
    reactions: Mapped[Optional[list[Reaction]]] = relationship(
        back_populates="message",
        enable_typechecks=False,
    )
    poll: Mapped[Optional[Poll]] = relationship(
        back_populates="message",
        enable_typechecks=False,
    )

    # The date the message was sent
    date: Mapped[Optional[timestamp]]
    # Type of the message
    type: Mapped[Optional[str_256]]
    # Number of views the message has (this can be manipulated)
    views: Mapped[Optional[bigint_u]]
    # Number of times the message was forwarded
    forwards: Mapped[Optional[bigint_u]]
    # For messages forwarded from users who have hidden their accounts, name of the user.
    forward_sender_name: Mapped[Optional[str_256]]
    # The date the message was originally sent
    forward_date: Mapped[Optional[timestamp]]

    """ Proxies
        --------
        These are the relationships to the other tables.
    """
    from_user: Mapped[Optional[User]] = relationship(
        "User",
        remote_side=[User.id],
        foreign_keys=[from_user_id],
    )
    sender_chat: Mapped[Optional[Chat]] = relationship(
        "Chat",
        remote_side=[Chat.id],
        foreign_keys=[sender_chat_id],
        enable_typechecks=False,
    )
    forward_from: Mapped[Optional[User]] = relationship(
        "User",
        remote_side=[User.id],
        foreign_keys=[forward_from_id],
        enable_typechecks=False,
    )
    forward_from_chat: Mapped[Optional[Chat]] = relationship(
        "Chat",
        remote_side=[Chat.id],
        foreign_keys=[forward_from_chat_id],
        enable_typechecks=False,
    )
    chat: Mapped[Optional[Chat]] = relationship(
        "Chat",
        remote_side=[Chat.id],
        foreign_keys=[chat_id],
        enable_typechecks=False,
    )
    reply_to_message: Mapped[Optional[Message]] = relationship(
        "Message",
        remote_side=[id],
        foreign_keys=[reply_to_message_id],
        enable_typechecks=False,
    )
    reply_to_top_message: Mapped[Optional[Message]] = relationship(
        "Message",
        remote_side=[id],
        foreign_keys=[reply_to_top_message_id],
        enable_typechecks=False,
    )

    def __repr__(self):  # pragma: no cover
        """Return a string representation of the Message model."""
        return f"<Message '@{self.id}'>"


class MessageTextContent(Base):

    __tablename__ = "message_content"

    # Mapping to the message (one-to-one relationship)
    message_id: Mapped[bigint_u] = mapped_column(ForeignKey("messages.id"), primary_key=True)
    message: Mapped[Message] = relationship(
        remote_side=[Message.id], foreign_keys=[message_id], single_parent=True
    )

    # ---------------------------------------------------------------------------- #
    #                                    content                                   #
    # ---------------------------------------------------------------------------- #

    # The text of the message, if it's a text message should be max 4096 characters
    text: Mapped[Optional[str_4096]]
    # The caption of the message, if it's a media message ~~should be max 1024 characters~~
    caption: Mapped[Optional[str_4096]]
    # If the type is a sticker, the emoji of the sticker
    sticker_emoji: Mapped[Optional[str_1024]]
    sticker_set_name: Mapped[Optional[str_1024]]
    sticker_file_unique_id: Mapped[Optional[str_1024]]

    def __repr__(self):  # pragma: no cover
        """Return a string representation of the MessageTextContent model."""
        return f"<MessageTextContent '@{self.message_id}'>"


class EntityHashtag(Base):
    __tablename__ = 'entity_hashtags'
    __bind_key__ = "Base"

    entity_id: Mapped[bigint_u] = mapped_column(primary_key=True)
    message_id: Mapped[bigint_u] = mapped_column(
        ForeignKey("messages.id"), index=True, nullable=False
    )
    hashtag: Mapped[str] = mapped_column(nullable=False)

    message: Mapped[Message] = relationship(
        remote_side=[Message.id],
        foreign_keys=[message_id],
        enable_typechecks=False,
    )

    def __repr__(self):  # pragma: no cover
        return f"<EntityHashtag '@{self.entity_id}'>"


class EntityURL(Base):
    __tablename__ = 'entity_urls'
    __bind_key__ = "Base"

    entity_id: Mapped[bigint_u] = mapped_column(primary_key=True)
    message_id: Mapped[bigint_u] = mapped_column(
        ForeignKey("messages.id"), index=True, nullable=False
    )
    url: Mapped[str] = mapped_column(nullable=False)

    message: Mapped[Message] = relationship(
        remote_side=[Message.id],
        foreign_keys=[message_id],
        enable_typechecks=False,
    )

    def __repr__(self):  # pragma: no cover
        return f"<EntityURL '@{self.entity_id}'>"


class Reaction(Base):
    """A reaction on the message.

    It can be any emoji, for instance a thumbs up
    or a heart. It also contains the count of how many times it was used. The order
    is the order in which the reactions are displayed in the message.
    Sometimes custom emojis are used, in that case the emoji string can not be parsed
    easily.
    """

    __tablename__ = "reactions"

    # The internal identifier of each reaction (one-to-many relationship to message)
    id: Mapped[bigint_u] = mapped_column(primary_key=True, autoincrement=True)
    message_id: Mapped[bigint_u] = mapped_column(ForeignKey("messages.id"))
    message: Mapped[Message] = relationship(
        remote_side=[Message.id],
        foreign_keys=[message_id],
        enable_typechecks=False,
    )

    emoji: Mapped[Optional[str_256]]  # sometimes null i think if custom emoji
    count: Mapped[int]
    order: Mapped[int]

    def __repr__(self):  # pragma: no cover
        return f"<Reaction '@{self.id}'>"
