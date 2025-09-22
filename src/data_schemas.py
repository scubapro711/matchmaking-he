"""
סכימות נתונים ולוולידציה למערכת השידוכים
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime

class Gender(str, Enum):
    MALE = "M"
    FEMALE = "F"

class MaritalStatus(str, Enum):
    SINGLE = "single"
    DIVORCED = "divorced"
    WIDOWED = "widowed"

class Community(str, Enum):
    LITHUANIAN = "lithuanian"
    HASIDIC = "hasidic"
    SEPHARDIC = "sephardic"
    MODERN_ORTHODOX = "modern_orthodox"
    NATIONAL_RELIGIOUS = "national_religious"

class ReligiosityLevel(str, Enum):
    VERY_STRICT = "very_strict"
    STRICT = "strict"
    MODERATE = "moderate"
    FLEXIBLE = "flexible"

class EducationLevel(str, Enum):
    HIGH_SCHOOL = "high_school"
    SEMINARY = "seminary"
    YESHIVA = "yeshiva"
    COLLEGE = "college"
    UNIVERSITY = "university"
    ADVANCED_DEGREE = "advanced_degree"

class HardConstraints(BaseModel):
    """תנאי סף מחייבים"""
    min_age: Optional[int] = Field(None, ge=18, le=120)
    max_age: Optional[int] = Field(None, ge=18, le=120)
    max_distance_km: Optional[int] = Field(None, ge=0, le=1000)
    smoking: Optional[bool] = None
    required_communities: Optional[List[Community]] = None
    required_religiosity: Optional[List[ReligiosityLevel]] = None
    required_languages: Optional[List[str]] = None

    @validator('max_age')
    def validate_age_range(cls, v, values):
        if 'min_age' in values and values['min_age'] and v:
            if v < values['min_age']:
                raise ValueError('max_age must be greater than min_age')
        return v

class NiceToHave(BaseModel):
    """העדפות שמעלות ציון"""
    preferred_education: Optional[List[EducationLevel]] = None
    preferred_occupation: Optional[List[str]] = None
    preferred_location: Optional[str] = None
    family_size_preference: Optional[str] = None
    lifestyle_preferences: Optional[List[str]] = None

class Candidate(BaseModel):
    source: Optional[str] = None  # To track the origin of the data
    """מועמד/ת לשידוך"""
    id: str = Field(..., description="מזהה ייחודי")
    gender: Gender
    age: int = Field(..., ge=18, le=120)
    marital_status: MaritalStatus
    community: Community
    religiosity_level: ReligiosityLevel
    location: str = Field(..., description="עיר/אזור מגורים")
    education: EducationLevel
    occupation: Optional[str] = None
    description_text: str = Field(..., description="תיאור אישי חופשי")
    languages: List[str] = Field(default=["hebrew"])
    smoking: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    
    # נתונים רגישים - מוצפנים
    full_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

class Preferences(BaseModel):
    """העדפות מועמד/ת"""
    candidate_id: str
    must_have: HardConstraints
    nice_to_have: NiceToHave
    free_text: str = Field(..., description="תיאור חופשי של העדפות")
    updated_at: datetime = Field(default_factory=datetime.now)

class MatchScore(BaseModel):
    """ציון התאמה בין שני מועמדים"""
    candidate_a_id: str
    candidate_b_id: str
    total_score: float = Field(..., ge=0, le=1)
    semantic_similarity: float = Field(..., ge=0, le=1)
    religious_compatibility: float = Field(..., ge=0, le=1)
    age_compatibility: float = Field(..., ge=0, le=1)
    location_compatibility: float = Field(..., ge=0, le=1)
    other_factors: float = Field(..., ge=0, le=1)
    explanation: str = Field(..., description="הסבר על הציון")
    calculated_at: datetime = Field(default_factory=datetime.now)

class FeedbackStatus(str, Enum):
    SENT = "sent"
    CONTACT_MADE = "contact_made"
    MEETING_ARRANGED = "meeting_arranged"
    MATCHED = "matched"
    REJECTED = "rejected"
    NO_RESPONSE = "no_response"

class Feedback(BaseModel):
    """פידבק על התאמה"""
    candidate_a_id: str
    candidate_b_id: str
    status: FeedbackStatus
    reason: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    timestamp: datetime = Field(default_factory=datetime.now)

class MatchRequest(BaseModel):
    """בקשת חיפוש התאמות"""
    candidate_id: str
    max_results: int = Field(20, ge=1, le=100)
    min_score: float = Field(0.3, ge=0, le=1)
    include_explanation: bool = True

class MatchResponse(BaseModel):
    """תגובת חיפוש התאמות"""
    candidate_id: str
    matches: List[MatchScore]
    total_found: int
    search_time_ms: int
