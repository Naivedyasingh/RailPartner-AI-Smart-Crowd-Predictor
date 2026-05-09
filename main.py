from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr, field_validator
from pydantic_settings import BaseSettings
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from contextlib import asynccontextmanager
from functools import lru_cache
import numpy as np
import joblib
import os
import logging
import secrets
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    DATABASE_URL: str =os.getenv('DATABASE_URL')
    SECRET_KEY: str =os.getenv('SECRET_KEY')
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALLOWED_ORIGINS: str = "http://localhost:8501"

    @property
    def origins_list(self):
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"sslmode": "require"},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, index=True, nullable=False)
    username        = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    predictions     = relationship("PredictionHistory", back_populates="user")


class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_weekend       = Column(Boolean)
    holiday_type     = Column(Integer)
    distance         = Column(Float)
    month            = Column(Integer)
    journey_type     = Column(String)
    sl_capacity      = Column(Integer)
    ac3_capacity     = Column(Integer)
    sl_booked        = Column(Integer)
    ac3_booked       = Column(Integer)
    ac2_booked       = Column(Integer)
    ac1_booked       = Column(Integer)
    ac2_capacity     = Column(Integer)
    ac1_capacity     = Column(Integer)
    crowd_level      = Column(String)
    crowd_confidence = Column(Float)
    seat_status      = Column(String)
    seat_confidence  = Column(Float)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())
    user             = relationship("User", back_populates="predictions")


pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({**data, "exp": expire}, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or disabled")
    return user


class UserRegister(BaseModel):
    email: EmailStr
    username: str
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

class UserOut(BaseModel):
    id: int
    email: str
    username: str
    is_active: bool
    created_at: datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

class PredictionInput(BaseModel):
    is_weekend:   bool
    holiday_type: int
    distance:     float
    month:        int
    journey_type: str
    sl_capacity:  int
    ac3_capacity: int
    sl_booked:    int = 0
    ac3_booked:   int = 0
    ac2_booked:   int = 0
    ac1_booked:   int = 0
    ac2_capacity: int = 0
    ac1_capacity: int = 0

    @field_validator("month")
    @classmethod
    def valid_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError("month must be 1–12")
        return v

class ModelResult(BaseModel):
    label:         str
    confidence:    float
    probabilities: dict

class PredictionResponse(BaseModel):
    crowd: ModelResult
    seat:  ModelResult

class HistoryOut(BaseModel):
    id:               int
    crowd_level:      str
    crowd_confidence: float
    seat_status:      str
    seat_confidence:  float
    journey_type:     Optional[str]
    distance:         float
    created_at:       datetime
    class Config:
        from_attributes = True


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

class ModelRegistry:
    def __init__(self):
        self.crowd_model   = None
        self.seat_model    = None
        self.crowd_encoder = None
        self.seat_encoder  = None
        self._loaded       = False

    def load(self):
        if self._loaded:
            return
        try:
            self.crowd_model   = joblib.load(os.path.join(MODELS_DIR, "crowd_model.pkl"))
            self.crowd_encoder = joblib.load(os.path.join(MODELS_DIR, "CROWD_encoder.pkl"))
            self.seat_model    = joblib.load(os.path.join(MODELS_DIR, "seat_model.pkl"))
            self.seat_encoder  = joblib.load(os.path.join(MODELS_DIR, "SEAT_encoder.pkl"))
            self._loaded = True
            logger.info("✅ All 4 ML models loaded successfully")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"❌ Model file missing: {e}\n"
                f"Place all 4 .pkl files inside: {MODELS_DIR}"
            )

    def _encode_journey(self, journey_type: str) -> int:
        mapping = {
            "express": 0, "superfast": 1, "local": 2,
            "mail": 3, "intercity": 4, "rajdhani": 5,
        }
        return mapping.get(str(journey_type).lower(), 0)

    def predict_crowd(self, f: dict) -> dict:
        X = np.array([[
            int(f["is_weekend"]),
            f["holiday_type"],
            f["distance"],
            f["month"],
            self._encode_journey(f["journey_type"]),
            f["sl_capacity"],
            f["ac3_capacity"],
        ]])
        probs     = self.crowd_model.predict(X)[0]
        class_idx = int(np.argmax(probs))
        label     = self.crowd_encoder.inverse_transform([class_idx])[0]
        return {
            "label":         label,
            "confidence":    float(probs[class_idx]),
            "probabilities": {c: float(p) for c, p in zip(self.crowd_encoder.classes_, probs)},
        }

    def predict_seat(self, f: dict) -> dict:
        total_booked     = f["sl_booked"] + f["ac3_booked"] + f["ac2_booked"] + f["ac1_booked"]
        total_cap        = f["sl_capacity"] + f["ac3_capacity"] + f["ac2_capacity"] + f["ac1_capacity"]
        booking_ratio    = min(total_booked / (total_cap + 1), 0.95)
        demand_pressure  = (booking_ratio * 0.7
                            + int(f["is_weekend"]) * 0.15
                            + f["holiday_type"] * 0.15)
        sl_pressure      = f["sl_booked"]  / (f["sl_capacity"]  + 1)
        ac3_pressure     = f["ac3_booked"] / (f["ac3_capacity"] + 1)

        X = np.array([[
            booking_ratio, demand_pressure, sl_pressure, ac3_pressure,
            int(f["is_weekend"]), f["holiday_type"],
            self._encode_journey(f["journey_type"]), f["month"], 0,
        ]])
        probs     = self.seat_model.predict_proba(X)[0]
        class_idx = int(np.argmax(probs))
        label     = self.seat_encoder.inverse_transform([class_idx])[0]
        return {
            "label":         label,
            "confidence":    float(probs[class_idx]),
            "probabilities": {c: float(p) for c, p in zip(self.seat_encoder.classes_, probs)},
        }

@lru_cache()
def get_registry():
    reg = ModelRegistry()
    reg.load()
    return reg


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting — creating DB tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Tables ready")
    get_registry()       
    yield
    logger.info("👋 Shutdown complete")

app = FastAPI(
    title="Train Predictor API",
    version="1.0.0",
    description="Crowd & seat availability prediction with JWT auth",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/auth/register", response_model=UserOut, status_code=201, tags=["Auth"])
def register(payload: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(409, "Email already registered")
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(409, "Username already taken")
    user = User(
        email=payload.email,
        username=payload.username,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/api/auth/login", response_model=Token, tags=["Auth"])
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer", "user": user}


@app.get("/api/auth/me", response_model=UserOut, tags=["Auth"])
def me(current_user: User = Depends(get_current_user)):
    return current_user


@app.post("/api/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(
    payload: PredictionInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    registry = get_registry()
    features = payload.model_dump()

    try:
        crowd = registry.predict_crowd(features)
        seat  = registry.predict_seat(features)
    except Exception as e:
        raise HTTPException(500, f"Model inference error: {e}")

    db.add(PredictionHistory(
        user_id          = current_user.id,
        is_weekend       = payload.is_weekend,
        holiday_type     = payload.holiday_type,
        distance         = payload.distance,
        month            = payload.month,
        journey_type     = payload.journey_type,
        sl_capacity      = payload.sl_capacity,
        ac3_capacity     = payload.ac3_capacity,
        sl_booked        = payload.sl_booked,
        ac3_booked       = payload.ac3_booked,
        ac2_booked       = payload.ac2_booked,
        ac1_booked       = payload.ac1_booked,
        ac2_capacity     = payload.ac2_capacity,
        ac1_capacity     = payload.ac1_capacity,
        crowd_level      = crowd["label"],
        crowd_confidence = crowd["confidence"],
        seat_status      = seat["label"],
        seat_confidence  = seat["confidence"],
    ))
    db.commit()

    return {"crowd": crowd, "seat": seat}


@app.get("/api/predict/history", response_model=List[HistoryOut], tags=["Predictions"])
def history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return (
        db.query(PredictionHistory)
        .filter(PredictionHistory.user_id == current_user.id)
        .order_by(PredictionHistory.created_at.desc())
        .limit(limit)
        .all()
    )


@app.delete("/api/predict/history/{record_id}", status_code=204, tags=["Predictions"])
def delete_record(
    record_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = db.query(PredictionHistory).filter(
        PredictionHistory.id == record_id,
        PredictionHistory.user_id == current_user.id,
    ).first()
    if not rec:
        raise HTTPException(404, "Record not found")
    db.delete(rec)
    db.commit()


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "app": "Train Predictor API"}