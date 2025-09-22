"""
FastAPI למערכת השידוכים ההיברידית
"""
import time
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from src.data_schemas import (
    Candidate, Preferences, MatchRequest, MatchResponse, 
    MatchScore, Feedback, FeedbackStatus
)
from src.rules_filter import RulesFilter
from src.embeddings import HebrewEmbeddings
from src.scoring import MatchingScorer
from src.ranker import MatchingRanker
from src.stable_matching import StableMatching

# הגדרת לוגינג
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# יצירת אפליקציית FastAPI
app = FastAPI(
    title="מערכת שידוכים היברידית",
    description="מערכת שידוכים מתקדמת עם AI - כללים, אמבדינגים ולמידת דירוג",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# הגדרת CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # בפרודקשן - להגביל לדומיינים ספציפיים
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# אבטחה בסיסית
security = HTTPBearer()

# משתנים גלובליים למערכת
rules_filter = RulesFilter()
embeddings = HebrewEmbeddings()
scorer = MatchingScorer()
ranker = MatchingRanker()
stable_matcher = StableMatching()

# מאגרי נתונים זמניים (בפרודקשן - להחליף בבסיס נתונים)
import pandas as pd

candidates_db: Dict[str, Candidate] = {}

@app.on_event("startup")
async def startup_event():
    """טעינת נתונים ראשונית"""
    try:
        df = pd.read_csv("data/unified_candidates.csv")
        for _, row in df.iterrows():
            candidate = Candidate(
                id=row["id"],
                gender=row["gender"],
                age=row["age"],
                marital_status=row["marital_status"],
                community=row["community"],
                religiosity_level=row["religiosity_level"],
                location=row["location"],
                education=row["education"],
                occupation=row["occupation"],
                description_text=row["description_text"],
                languages=row["languages"].split(",") if isinstance(row["languages"], str) else ["hebrew"],
                smoking=row["smoking"],
                source=row["source"]
            )
            candidates_db[candidate.id] = candidate
        logger.info(f"Loaded {len(candidates_db)} candidates from unified_candidates.csv")
    except FileNotFoundError:
        logger.warning("unified_candidates.csv not found. Starting with an empty database.")
    except Exception as e:
        logger.error(f"Failed to load initial data: {e}")
preferences_db: Dict[str, Preferences] = {}
feedback_db: List[Feedback] = []

class SystemStatus(BaseModel):
    """סטטוס המערכת"""
    status: str
    components: Dict[str, bool]
    candidates_count: int
    feedback_count: int
    ranker_trained: bool

class BatchMatchRequest(BaseModel):
    """בקשת התאמות מרובות"""
    candidate_ids: List[str]
    max_results_per_candidate: int = 10
    min_score: float = 0.3
    use_stable_matching: bool = False

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """אימות טוקן (פשוט לדוגמה)"""
    # בפרודקשן - להחליף באימות אמיתי
    if credentials.credentials != "demo_token_123":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

@app.get("/", response_model=Dict[str, str])
async def root():
    """נקודת כניסה ראשית"""
    return {
        "message": "ברוכים הבאים למערכת השידוכים ההיברידית",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """בדיקת תקינות המערכת"""
    components = {
        "rules_filter": rules_filter is not None,
        "embeddings": embeddings.model is not None,
        "scorer": scorer is not None,
        "ranker": ranker is not None,
        "stable_matcher": stable_matcher is not None
    }
    
    return SystemStatus(
        status="healthy" if all(components.values()) else "degraded",
        components=components,
        candidates_count=len(candidates_db),
        feedback_count=len(feedback_db),
        ranker_trained=ranker.is_trained
    )

@app.post("/candidates", response_model=Dict[str, str])
async def add_candidate(candidate: Candidate, token: str = Depends(verify_token)):
    """הוספת מועמד חדש"""
    try:
        candidates_db[candidate.id] = candidate
        logger.info(f"Added candidate: {candidate.id}")
        return {"message": f"מועמד {candidate.id} נוסף בהצלחה", "candidate_id": candidate.id}
    except Exception as e:
        logger.error(f"Failed to add candidate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidates/{candidate_id}", response_model=Candidate)
async def get_candidate(candidate_id: str, token: str = Depends(verify_token)):
    """קבלת פרטי מועמד"""
    if candidate_id not in candidates_db:
        raise HTTPException(status_code=404, detail="מועמד לא נמצא")
    
    candidate = candidates_db[candidate_id]
    # הסתרת מידע רגיש
    candidate.full_name = None
    candidate.phone = None
    candidate.email = None
    
    return candidate

@app.post("/preferences", response_model=Dict[str, str])
async def add_preferences(preferences: Preferences, token: str = Depends(verify_token)):
    """הוספת העדפות מועמד"""
    try:
        if preferences.candidate_id not in candidates_db:
            raise HTTPException(status_code=404, detail="מועמד לא נמצא")
        
        preferences_db[preferences.candidate_id] = preferences
        logger.info(f"Added preferences for candidate: {preferences.candidate_id}")
        return {"message": f"העדפות למועמד {preferences.candidate_id} נוספו בהצלחה"}
    except Exception as e:
        logger.error(f"Failed to add preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match/search", response_model=MatchResponse)
async def search_matches(request: MatchRequest, token: str = Depends(verify_token)):
    """חיפוש התאמות למועמד"""
    start_time = time.time()
    
    try:
        # בדיקת קיום המועמד
        if request.candidate_id not in candidates_db:
            raise HTTPException(status_code=404, detail="מועמד לא נמצא")
        
        if request.candidate_id not in preferences_db:
            raise HTTPException(status_code=404, detail="העדפות למועמד לא נמצאו")
        
        requester = candidates_db[request.candidate_id]
        requester_preferences = preferences_db[request.candidate_id]
        
        # סינון מועמדים לפי תנאי סף
        all_candidates = list(candidates_db.values())
        valid_candidates = rules_filter.filter_candidates(
            all_candidates, requester, requester_preferences
        )
        
        if not valid_candidates:
            return MatchResponse(
                candidate_id=request.candidate_id,
                matches=[],
                total_found=0,
                search_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # חישוב ציונים
        matches = []
        for candidate, _ in valid_candidates:
            if candidate.id not in preferences_db:
                continue
            
            candidate_preferences = preferences_db[candidate.id]
            
            # חישוב ציון בסיסי
            base_score = scorer.calculate_match_score(
                requester, candidate, requester_preferences, candidate_preferences
            )
            
            # שיפור עם ranker אם זמין
            if ranker.is_trained:
                base_scores_dict = {
                    'total_score': base_score.total_score,
                    'semantic_similarity': base_score.semantic_similarity,
                    'religious_compatibility': base_score.religious_compatibility,
                    'age_compatibility': base_score.age_compatibility,
                    'location_compatibility': base_score.location_compatibility,
                    'other_factors': base_score.other_factors
                }
                
                enhanced_score = ranker.predict_score(
                    requester, candidate, requester_preferences, 
                    candidate_preferences, base_scores_dict
                )
                
                base_score.total_score = enhanced_score
            
            if base_score.total_score >= request.min_score:
                matches.append(base_score)
        
        # מיון ומגבלה
        matches.sort(key=lambda x: x.total_score, reverse=True)
        matches = matches[:request.max_results]
        
        search_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"Found {len(matches)} matches for {request.candidate_id} in {search_time}ms")
        
        return MatchResponse(
            candidate_id=request.candidate_id,
            matches=matches,
            total_found=len(matches),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Failed to search matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match/batch", response_model=Dict[str, List[MatchScore]])
async def batch_search_matches(request: BatchMatchRequest, token: str = Depends(verify_token)):
    """חיפוש התאמות מרובות"""
    try:
        results = {}
        
        for candidate_id in request.candidate_ids:
            match_request = MatchRequest(
                candidate_id=candidate_id,
                max_results=request.max_results_per_candidate,
                min_score=request.min_score
            )
            
            response = await search_matches(match_request, token)
            results[candidate_id] = response.matches
        
        # התאמות יציבות אופציונליות
        if request.use_stable_matching and len(request.candidate_ids) > 1:
            all_scores = []
            for matches in results.values():
                all_scores.extend(matches)
            
            stable_matches = stable_matcher.find_stable_matches(
                all_scores, candidates_db, request.min_score
            )
            
            # עדכון התוצאות עם התאמות יציבות
            results["stable_matches"] = stable_matches
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to batch search matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", response_model=Dict[str, str])
async def add_feedback(feedback: Feedback, token: str = Depends(verify_token)):
    """הוספת פידבק על התאמה"""
    try:
        # בדיקת קיום המועמדים
        if feedback.candidate_a_id not in candidates_db:
            raise HTTPException(status_code=404, detail="מועמד A לא נמצא")
        
        if feedback.candidate_b_id not in candidates_db:
            raise HTTPException(status_code=404, detail="מועמד B לא נמצא")
        
        feedback_db.append(feedback)
        
        logger.info(f"Added feedback: {feedback.candidate_a_id} -> {feedback.candidate_b_id} "
                   f"({feedback.status.value})")
        
        return {"message": "פידבק נוסף בהצלחה"}
        
    except Exception as e:
        logger.error(f"Failed to add feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ranker/train", response_model=Dict[str, str])
async def train_ranker(token: str = Depends(verify_token)):
    """אימון מודל הדירוג"""
    try:
        if len(feedback_db) < 10:
            raise HTTPException(
                status_code=400, 
                detail=f"לא מספיק פידבק לאימון. נדרשים לפחות 10, יש {len(feedback_db)}"
            )
        
        # חישוב ציונים בסיסיים לכל הפידבק
        base_scores_data = {}
        
        for feedback in feedback_db:
            candidate_a = candidates_db.get(feedback.candidate_a_id)
            candidate_b = candidates_db.get(feedback.candidate_b_id)
            preferences_a = preferences_db.get(feedback.candidate_a_id)
            preferences_b = preferences_db.get(feedback.candidate_b_id)
            
            if all([candidate_a, candidate_b, preferences_a, preferences_b]):
                base_score = scorer.calculate_match_score(
                    candidate_a, candidate_b, preferences_a, preferences_b
                )
                
                base_scores_data[(feedback.candidate_a_id, feedback.candidate_b_id)] = {
                    'total_score': base_score.total_score,
                    'semantic_similarity': base_score.semantic_similarity,
                    'religious_compatibility': base_score.religious_compatibility,
                    'age_compatibility': base_score.age_compatibility,
                    'location_compatibility': base_score.location_compatibility,
                    'other_factors': base_score.other_factors
                }
        
        # אימון המודל
        ranker.train(feedback_db, candidates_db, preferences_db, base_scores_data)
        
        return {"message": f"מודל הדירוג אומן בהצלחה על {len(feedback_db)} דוגמאות פידבק"}
        
    except Exception as e:
        logger.error(f"Failed to train ranker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ranker/importance", response_model=Dict[str, float])
async def get_feature_importance(token: str = Depends(verify_token)):
    """קבלת חשיבות הפיצ'רים במודל הדירוג"""
    if not ranker.is_trained:
        raise HTTPException(status_code=400, detail="מודל הדירוג לא אומן עדיין")
    
    return ranker.get_feature_importance()

@app.get("/stats", response_model=Dict[str, int])
async def get_statistics(token: str = Depends(verify_token)):
    """סטטיסטיקות המערכת"""
    feedback_by_status = {}
    for feedback in feedback_db:
        status = feedback.status.value
        feedback_by_status[status] = feedback_by_status.get(status, 0) + 1
    
    return {
        "total_candidates": len(candidates_db),
        "total_preferences": len(preferences_db),
        "total_feedback": len(feedback_db),
        "ranker_trained": ranker.is_trained,
        **feedback_by_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
