"""
Database connection setup.

DATABASE_URL env var controls the target:
  - unset -> sqlite:///./upi_app.db  (zero-setup local dev/testing)
  - postgresql://user:pass@host:5432/dbname  -> real Postgres, no code
    changes needed anywhere else in the app; SQLAlchemy handles the
    dialect difference.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./upi_app.db")

# Newer SQLAlchemy reads a plain "postgresql://" URL as the psycopg v3 driver,
# but requirements.txt installs psycopg2. Name the driver explicitly so
# "postgresql://user:pass@host/db" just works.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg2://" + DATABASE_URL[len("postgresql://"):]

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from . import models  # noqa: F401  (ensures models are registered on Base before create_all)
    Base.metadata.create_all(bind=engine)
