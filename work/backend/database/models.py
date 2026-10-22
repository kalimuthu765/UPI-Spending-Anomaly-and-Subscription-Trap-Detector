"""
Phase 9 — Database Schema (SQLAlchemy ORM)
Works against SQLite out of the box (zero setup, for local dev/testing)
and against real PostgreSQL by just setting DATABASE_URL — no code
changes, since SQLAlchemy abstracts the dialect.

Tables (per the spec): users, user_profiles, transactions, subscriptions,
anomaly_alerts, model_versions. (merchants/categories are kept as plain
string columns rather than separate lookup tables — this project's
merchant/category sets are open-ended, not a fixed catalog, so a lookup
table would just add joins without adding integrity value.)
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
)
from sqlalchemy.orm import relationship
from .db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    profile = relationship("UserProfile", back_populates="user", uselist=False)
    transactions = relationship("Transaction", back_populates="user")
    subscriptions = relationship("Subscription", back_populates="user")
    alerts = relationship("AnomalyAlert", back_populates="user")


class UserProfile(Base):
    """The per-user spending baseline that makes personalization persist
    across logins instead of being recomputed from scratch every upload."""
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    total_transactions = Column(Integer, default=0)
    average_spending = Column(Float, default=0.0)
    median_spending = Column(Float, default=0.0)
    spending_std = Column(Float, default=0.0)
    has_personal_model = Column(Boolean, default=False)   # True once txn count crosses the threshold
    personal_model_path = Column(String, nullable=True)
    trained_on_count = Column(Integer, default=0)          # txn count the SAVED model was trained on
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="profile")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    transaction_id = Column(String, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    merchant = Column(String, nullable=False, index=True)
    category = Column(String, nullable=False, index=True)
    payment_mode = Column(String, nullable=True)

    user = relationship("User", back_populates="transactions")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    merchant = Column(String, nullable=False)
    avg_amount = Column(Float, nullable=False)
    cadence = Column(String, nullable=True)               # monthly / weekly / annual
    recurring_probability = Column(Float, nullable=True)
    last_charge = Column(DateTime, nullable=True)
    next_expected_payment = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    user = relationship("User", back_populates="subscriptions")


class AnomalyAlert(Base):
    __tablename__ = "anomaly_alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    transaction_id = Column(String, index=True)
    timestamp = Column(DateTime, nullable=False)
    amount = Column(Float, nullable=False)
    merchant = Column(String, nullable=True)
    anomaly_score = Column(Float, nullable=False)
    explanation = Column(Text, nullable=True)
    model_used = Column(String, nullable=True)            # 'global' or 'personal'
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="alerts")


class ModelVersion(Base):
    """Mirrors what's tracked in MLflow, for quick lookup without querying
    the MLflow store from the API layer. Also used for per-user personal
    model versions (Phase 12 retraining pipeline)."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # null = global model
    name = Column(String, nullable=False)                 # e.g. UPI_Anomaly_Model
    version = Column(String, nullable=False)
    scope = Column(String, nullable=True)                 # global / personalized
    trained_on_count = Column(Integer, nullable=True)      # how many transactions it was trained on
    f1 = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    registered_at = Column(DateTime, default=datetime.utcnow)


class DriftReport(Base):
    """Phase 11 — logged on every upload once a user has a prior baseline
    to compare against, whether or not drift crossed the retrain threshold."""
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    psi = Column(Float, nullable=True)
    ks_statistic = Column(Float, nullable=True)
    ks_pvalue = Column(Float, nullable=True)
    baseline_mean = Column(Float, nullable=True)
    current_mean = Column(Float, nullable=True)
    severity = Column(String, nullable=True)              # none / moderate / significant
    drift_detected = Column(Boolean, default=False)
    retrain_triggered = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Credit(Base):
    """Money that came in (salary, refunds, transfers received). Separate from
    transactions on purpose: spending analysis never sees these rows."""
    __tablename__ = "credits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    source = Column(String, nullable=False, index=True)
    kind = Column(String, nullable=False, default="Credit")       # "Credit" | "Refund"
