import decimal

from sqlalchemy import TEXT, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, registry
from typing_extensions import Annotated
from abc import ABCMeta

from sqlalchemy.dialects import postgresql

# Type declaration for own use
bigint_u = Annotated[int, "unsigned"]
bigint = Annotated[int, "signed"]
str_8 = Annotated[str, 8]
str_256 = Annotated[str, 256]
str_512 = Annotated[str, 512]
str_4096 = Annotated[str, 4096]
str_1024 = Annotated[str, 1024]
text = Annotated[str, "text"]
timestamp = TIMESTAMP
longtext = TEXT
double = Annotated[decimal.Decimal, "double"]
blob = Annotated[bytes, "blob"]

type_annotation_map = {
    int: postgresql.INTEGER(),
    bigint: postgresql.BIGINT(),
    # PostgreSQL handles large integers natively without an unsigned option
    bigint_u: postgresql.BIGINT(),
    # PostgreSQL allows specifying precision with fractional seconds
    timestamp: postgresql.TIMESTAMP(precision=6),
    double: postgresql.FLOAT(),  # PostgreSQL uses FLOAT for double-precision numbers
    # PostgreSQL supports native collation and character types with the database's default collation
    str_256: postgresql.VARCHAR(256),
    str_512: postgresql.VARCHAR(512),
    str_4096: postgresql.VARCHAR(4096),
    str_1024: postgresql.VARCHAR(1024),
    text: postgresql.TEXT(),
    longtext: postgresql.TEXT(),  # PostgreSQL doesn't differentiate between TEXT types like MySQL
    blob: postgresql.BYTEA(),  # PostgreSQL uses BYTEA for binary data
}


class Base(DeclarativeBase):
    """Base class for all models in the database."""

    __metaclass__ = ABCMeta  # make into abstract base class

    __abstract__ = True

    registry = registry(type_annotation_map=type_annotation_map)
