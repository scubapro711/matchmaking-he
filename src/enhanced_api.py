"""
API משופר למערכת השידוכים
מותאם לנתונים האמיתיים עם אלגוריתמים משופרים
"""
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import io
import json

from src.data_schemas import Candidate, Preferences, MatchScore
from src.enhanced_scoring import EnhancedMatchingScorer
from src.data_analyzer import DataAnalyzer
from src.rules_filter import RulesFilter
from src.ranker import LearningToRankModel

# הגדרת לוגינג
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# יצירת אפליקציה
app = FastAPI(
    title="מערכת שידוכים חכמה - Enhanced API",
    description="API משופר למערכת שידוכים היברידית מותאם לנתונים אמיתיים",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# הגדרת CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# אבטחה בסיסית
security = HTTPBearer()

# מופעי המחלקות
enhanced_scorer = EnhancedMatchingScorer()
data_analyzer = DataAnalyzer()
rules_filter = RulesFilter()
ranker = LearningToRankModel()

# מאגר נתונים זמני (במציאות יהיה מסד נתונים)
candidates_db: Dict[str, Dict] = {}
matches_history: List[Dict] = []

# מודלי נתונים לAPI
class CandidateInput(BaseModel):
    name: str
    age: int
    gender: str  # 'M' או 'F'
    height: Optional[float] = None
    location: str
    religious_sector: str
    institution: Optional[str] = None
    occupation: str
    preferences_text: str
    notes: Optional[str] = ""
    phone: Optional[str] = None

class MatchRequest(BaseModel):
    candidate_id: str
    max_results: int = 10
    min_score: float = 0.5
    include_explanation: bool = True

class FeedbackInput(BaseModel):
    candidate_a_id: str
    candidate_b_id: str
    outcome: str  # 'married', 'dated', 'talked', 'rejected'
    reason: Optional[str] = None

class BulkUploadResponse(BaseModel):
    success: bool
    processed_count: int
    errors: List[str]
    duplicates_removed: int
    data_quality_report: Dict[str, Any]

class AnalyticsResponse(BaseModel):
    total_candidates: int
    gender_distribution: Dict[str, int]
    age_distribution: Dict[str, float]
    religious_distribution: Dict[str, int]
    location_distribution: Dict[str, int]
    match_success_rate: float
    top_rejection_reasons: Dict[str, int]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """אימות טוקן פשוט (במציאות יהיה מורכב יותר)"""
    if credentials.credentials != "demo-token-2024":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

@app.get("/", tags=["General"])
async def root():
    """נקודת קצה בסיסית"""
    return {
        "message": "מערכת שידוכים חכמה - Enhanced API",
        "version": "2.0.0",
        "features": [
            "אלגוריתמים משופרים מבוססי נתונים אמיתיים",
            "ניתוח כפילויות ואיכות נתונים",
            "חילוץ העדפות אוטומטי מטקסט",
            "התאמת גובה ומרחק משופרת",
            "למידה מפידבק משתמשים"
        ],
        "docs": "/docs"
    }

@app.post("/candidates/bulk-upload", response_model=BulkUploadResponse, tags=["Candidates"])
async def bulk_upload_candidates(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """העלאה מרובה של מועמדים מקובץ CSV"""
    
    try:
        # קריאת הקובץ
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"Received file with {len(df)} records")
        
        # ניקוי וניתוח הנתונים
        cleaned_df = data_analyzer.load_and_clean_data_from_df(df)
        patterns = data_analyzer.analyze_matching_patterns(cleaned_df)
        
        # המרה למבנה הנתונים שלנו
        processed_count = 0
        errors = []
        
        for _, row in cleaned_df.iterrows():
            try:
                candidate_data = {
                    "id": str(row.get('מזהה', f"auto_{processed_count}")),
                    "name": f"{row.get('שם פרטי', '')} {row.get('שם משפחה', '')}".strip(),
                    "age": int(row.get('גיל_נומרי', 0)),
                    "gender": 'M' if row.get('מין') == 'זכר' else 'F',
                    "height": row.get('גובה נרמל'),
                    "location": row.get('עיר נרמל', ''),
                    "religious_sector": row.get('מגזר דתי נרמל', ''),
                    "institution": row.get('ישיבה/סמינר', ''),
                    "occupation": row.get('עיסוק', ''),
                    "preferences_text": str(row.get('הערות', '')),
                    "notes": str(row.get('הערות', '')),
                    "phone": row.get('טלפון נייד נקי'),
                    "marital_status": row.get('סטטוס משפחתי', ''),
                    "upload_date": datetime.now().isoformat()
                }
                
                candidates_db[candidate_data["id"]] = candidate_data
                processed_count += 1
                
            except Exception as e:
                errors.append(f"Row {processed_count}: {str(e)}")
        
        # דוח איכות נתונים
        quality_report = {
            "original_count": len(df),
            "processed_count": processed_count,
            "duplicates_removed": len(df) - len(cleaned_df),
            "missing_heights": cleaned_df['גובה נרמל'].isna().sum(),
            "missing_phones": cleaned_df['טלפון נייד נקי'].isna().sum(),
            "religious_distribution": cleaned_df['מגזר דתי נרמל'].value_counts().to_dict(),
            "age_range": {
                "min": float(cleaned_df['גיל_נומרי'].min()),
                "max": float(cleaned_df['גיל_נומרי'].max()),
                "mean": float(cleaned_df['גיל_נומרי'].mean())
            }
        }
        
        return BulkUploadResponse(
            success=True,
            processed_count=processed_count,
            errors=errors,
            duplicates_removed=len(df) - len(cleaned_df),
            data_quality_report=quality_report
        )
        
    except Exception as e:
        logger.error(f"Bulk upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/candidates/", tags=["Candidates"])
async def add_candidate(
    candidate: CandidateInput,
    token: str = Depends(verify_token)
):
    """הוספת מועמד יחיד"""
    
    candidate_id = f"manual_{len(candidates_db)}"
    
    candidate_data = {
        "id": candidate_id,
        "name": candidate.name,
        "age": candidate.age,
        "gender": candidate.gender,
        "height": candidate.height,
        "location": candidate.location,
        "religious_sector": candidate.religious_sector,
        "institution": candidate.institution,
        "occupation": candidate.occupation,
        "preferences_text": candidate.preferences_text,
        "notes": candidate.notes,
        "phone": candidate.phone,
        "upload_date": datetime.now().isoformat()
    }
    
    candidates_db[candidate_id] = candidate_data
    
    return {"message": "Candidate added successfully", "candidate_id": candidate_id}

@app.get("/candidates/{candidate_id}", tags=["Candidates"])
async def get_candidate(
    candidate_id: str,
    token: str = Depends(verify_token)
):
    """קבלת פרטי מועמד"""
    
    if candidate_id not in candidates_db:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    return candidates_db[candidate_id]

@app.get("/candidates/", tags=["Candidates"])
async def list_candidates(
    skip: int = 0,
    limit: int = 50,
    gender: Optional[str] = None,
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
    religious_sector: Optional[str] = None,
    location: Optional[str] = None,
    token: str = Depends(verify_token)
):
    """רשימת מועמדים עם סינון"""
    
    candidates = list(candidates_db.values())
    
    # סינונים
    if gender:
        candidates = [c for c in candidates if c.get('gender') == gender]
    
    if min_age:
        candidates = [c for c in candidates if c.get('age', 0) >= min_age]
    
    if max_age:
        candidates = [c for c in candidates if c.get('age', 0) <= max_age]
    
    if religious_sector:
        candidates = [c for c in candidates if religious_sector.lower() in c.get('religious_sector', '').lower()]
    
    if location:
        candidates = [c for c in candidates if location.lower() in c.get('location', '').lower()]
    
    # פגינציה
    total = len(candidates)
    candidates = candidates[skip:skip + limit]
    
    return {
        "candidates": candidates,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.post("/matches/find", tags=["Matching"])
async def find_matches(
    request: MatchRequest,
    token: str = Depends(verify_token)
):
    """חיפוש התאמות משופר"""
    
    if request.candidate_id not in candidates_db:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    source_candidate_data = candidates_db[request.candidate_id]
    
    # המרה לאובייקט Candidate
    source_candidate = Candidate(
        id=source_candidate_data["id"],
        name=source_candidate_data["name"],
        age=source_candidate_data["age"],
        gender=source_candidate_data["gender"],
        height=source_candidate_data.get("height"),
        location=source_candidate_data["location"],
        community=source_candidate_data.get("religious_sector"),
        occupation=source_candidate_data.get("occupation"),
        preferences=source_candidate_data.get("preferences_text", "")
    )
    
    matches = []
    
    # חיפוש בכל המועמדים האחרים
    for candidate_id, candidate_data in candidates_db.items():
        if candidate_id == request.candidate_id:
            continue
        
        # בדיקת מגדר מנוגד
        if candidate_data["gender"] == source_candidate_data["gender"]:
            continue
        
        # המרה לאובייקט Candidate
        target_candidate = Candidate(
            id=candidate_data["id"],
            name=candidate_data["name"],
            age=candidate_data["age"],
            gender=candidate_data["gender"],
            height=candidate_data.get("height"),
            location=candidate_data["location"],
            community=candidate_data.get("religious_sector"),
            occupation=candidate_data.get("occupation"),
            preferences=candidate_data.get("preferences_text", "")
        )
        
        # בדיקת כללים קשיחים
        if not rules_filter.passes_hard_rules(source_candidate, target_candidate):
            continue
        
        # חישוב ציון התאמה משופר
        match_score = enhanced_scorer.calculate_enhanced_match_score(
            source_candidate,
            target_candidate,
            notes_a=source_candidate_data.get("notes", ""),
            notes_b=candidate_data.get("notes", "")
        )
        
        if match_score.total_score >= request.min_score:
            match_result = {
                "candidate": candidate_data,
                "score": match_score.total_score,
                "compatibility_breakdown": {
                    "semantic_similarity": match_score.semantic_similarity,
                    "religious_compatibility": match_score.religious_compatibility,
                    "age_compatibility": match_score.age_compatibility,
                    "location_compatibility": match_score.location_compatibility,
                    "height_compatibility": match_score.other_factors
                }
            }
            
            if request.include_explanation:
                match_result["explanation"] = match_score.explanation
            
            matches.append(match_result)
    
    # מיון לפי ציון
    matches.sort(key=lambda x: x["score"], reverse=True)
    
    # הגבלת תוצאות
    matches = matches[:request.max_results]
    
    return {
        "source_candidate_id": request.candidate_id,
        "matches_found": len(matches),
        "matches": matches,
        "search_parameters": {
            "max_results": request.max_results,
            "min_score": request.min_score,
            "include_explanation": request.include_explanation
        }
    }

@app.post("/feedback/", tags=["Learning"])
async def submit_feedback(
    feedback: FeedbackInput,
    token: str = Depends(verify_token)
):
    """הגשת פידבק על התאמה"""
    
    # המרת תוצאה לציון מספרי
    outcome_scores = {
        'married': 5,
        'engaged': 4,
        'dated': 3,
        'talked': 2,
        'met': 1,
        'rejected': 0
    }
    
    feedback_record = {
        "candidate_a_id": feedback.candidate_a_id,
        "candidate_b_id": feedback.candidate_b_id,
        "outcome": feedback.outcome,
        "score": outcome_scores.get(feedback.outcome, 0),
        "reason": feedback.reason,
        "timestamp": datetime.now().isoformat()
    }
    
    matches_history.append(feedback_record)
    
    # עדכון מודל הלמידה (אם יש מספיק נתונים)
    if len(matches_history) >= 10:
        try:
            ranker.update_model_with_feedback(matches_history)
        except Exception as e:
            logger.warning(f"Failed to update learning model: {str(e)}")
    
    return {"message": "Feedback recorded successfully", "feedback_id": len(matches_history)}

@app.get("/analytics/", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_analytics(token: str = Depends(verify_token)):
    """קבלת אנליטיקה על המערכת"""
    
    if not candidates_db:
        return AnalyticsResponse(
            total_candidates=0,
            gender_distribution={},
            age_distribution={},
            religious_distribution={},
            location_distribution={},
            match_success_rate=0.0,
            top_rejection_reasons={}
        )
    
    candidates_list = list(candidates_db.values())
    
    # התפלגות מגדר
    gender_dist = {}
    for candidate in candidates_list:
        gender = candidate.get('gender', 'Unknown')
        gender_dist[gender] = gender_dist.get(gender, 0) + 1
    
    # התפלגות גיל
    ages = [c.get('age', 0) for c in candidates_list if c.get('age')]
    age_dist = {
        "min": min(ages) if ages else 0,
        "max": max(ages) if ages else 0,
        "mean": sum(ages) / len(ages) if ages else 0,
        "median": sorted(ages)[len(ages)//2] if ages else 0
    }
    
    # התפלגות מגזרים
    religious_dist = {}
    for candidate in candidates_list:
        sector = candidate.get('religious_sector', 'Unknown')
        religious_dist[sector] = religious_dist.get(sector, 0) + 1
    
    # התפלגות מיקומים
    location_dist = {}
    for candidate in candidates_list:
        location = candidate.get('location', 'Unknown')
        location_dist[location] = location_dist.get(location, 0) + 1
    
    # שיעור הצלחה
    if matches_history:
        successful_matches = len([m for m in matches_history if m.get('score', 0) >= 3])
        success_rate = successful_matches / len(matches_history)
    else:
        success_rate = 0.0
    
    # סיבות דחייה מובילות
    rejection_reasons = {}
    for match in matches_history:
        if match.get('outcome') == 'rejected' and match.get('reason'):
            reason = match['reason']
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    
    return AnalyticsResponse(
        total_candidates=len(candidates_list),
        gender_distribution=gender_dist,
        age_distribution=age_dist,
        religious_distribution=religious_dist,
        location_distribution=dict(sorted(location_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
        match_success_rate=success_rate,
        top_rejection_reasons=dict(sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:5])
    )

@app.get("/health", tags=["General"])
async def health_check():
    """בדיקת תקינות המערכת"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "candidates_count": len(candidates_db),
        "matches_history_count": len(matches_history),
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
