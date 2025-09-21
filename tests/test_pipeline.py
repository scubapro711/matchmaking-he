"""
בדיקות למערכת השידוכים ההיברידית
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from src.api import app
from src.data_schemas import (
    Candidate, Preferences, Gender, MaritalStatus, 
    Community, ReligiosityLevel, EducationLevel,
    HardConstraints, NiceToHave
)
from src.rules_filter import RulesFilter
from src.embeddings import HebrewEmbeddings
from src.scoring import MatchingScorer

# יצירת client לבדיקות
client = TestClient(app)

# נתוני בדיקה
TEST_TOKEN = "demo_token_123"
HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}

@pytest.fixture
def sample_candidate_male():
    """מועמד לדוגמה - גבר"""
    return Candidate(
        id="test_male_001",
        gender=Gender.MALE,
        age=28,
        marital_status=MaritalStatus.SINGLE,
        community=Community.LITHUANIAN,
        religiosity_level=ReligiosityLevel.STRICT,
        location="ירושלים",
        education=EducationLevel.UNIVERSITY,
        occupation="מהנדס תוכנה",
        description_text="אני אדם דתי ומשפחתי שאוהב ללמוד תורה ולחקור טכנולוגיה",
        languages=["hebrew", "english"],
        smoking=False
    )

@pytest.fixture
def sample_candidate_female():
    """מועמדת לדוגמה - אישה"""
    return Candidate(
        id="test_female_001",
        gender=Gender.FEMALE,
        age=25,
        marital_status=MaritalStatus.SINGLE,
        community=Community.LITHUANIAN,
        religiosity_level=ReligiosityLevel.STRICT,
        location="ירושלים",
        education=EducationLevel.SEMINARY,
        occupation="מורה",
        description_text="אני אישה דתייה ומשפחתית שאוהבת חינוך ועבודה עם ילדים",
        languages=["hebrew"],
        smoking=False
    )

@pytest.fixture
def sample_preferences():
    """העדפות לדוגמה"""
    return Preferences(
        candidate_id="test_male_001",
        must_have=HardConstraints(
            min_age=22,
            max_age=30,
            max_distance_km=50,
            smoking=False,
            required_communities=[Community.LITHUANIAN, Community.SEPHARDIC],
            required_religiosity=[ReligiosityLevel.STRICT, ReligiosityLevel.MODERATE]
        ),
        nice_to_have=NiceToHave(
            preferred_education=[EducationLevel.SEMINARY, EducationLevel.UNIVERSITY],
            preferred_location="ירושלים"
        ),
        free_text="מחפש בת זוג דתייה ומשפחתית שחולקת איתי את הערכים של תורה ומסורת"
    )

class TestAPI:
    """בדיקות API"""
    
    def test_health_check(self):
        """בדיקת תקינות המערכת"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
    
    def test_root_endpoint(self):
        """בדיקת נקודת כניסה ראשית"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_add_candidate(self, sample_candidate_male):
        """בדיקת הוספת מועמד"""
        response = client.post(
            "/candidates",
            json=sample_candidate_male.dict(),
            headers=HEADERS
        )
        assert response.status_code == 200
        data = response.json()
        assert "candidate_id" in data
        assert data["candidate_id"] == sample_candidate_male.id
    
    def test_add_candidate_unauthorized(self, sample_candidate_male):
        """בדיקת הוספת מועמד ללא הרשאה"""
        response = client.post(
            "/candidates",
            json=sample_candidate_male.dict()
        )
        assert response.status_code == 403
    
    def test_get_candidate(self, sample_candidate_male):
        """בדיקת קבלת פרטי מועמד"""
        # הוספת המועמד תחילה
        client.post(
            "/candidates",
            json=sample_candidate_male.dict(),
            headers=HEADERS
        )
        
        # קבלת פרטי המועמד
        response = client.get(
            f"/candidates/{sample_candidate_male.id}",
            headers=HEADERS
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_candidate_male.id
        assert data["gender"] == sample_candidate_male.gender.value
        # וידוא שמידע רגיש מוסתר
        assert data["full_name"] is None
        assert data["phone"] is None
        assert data["email"] is None
    
    def test_add_preferences(self, sample_candidate_male, sample_preferences):
        """בדיקת הוספת העדפות"""
        # הוספת המועמד תחילה
        client.post(
            "/candidates",
            json=sample_candidate_male.dict(),
            headers=HEADERS
        )
        
        # הוספת העדפות
        response = client.post(
            "/preferences",
            json=sample_preferences.dict(),
            headers=HEADERS
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

class TestRulesFilter:
    """בדיקות מסנן הכללים"""
    
    def test_age_constraint(self, sample_candidate_male, sample_candidate_female, sample_preferences):
        """בדיקת תנאי גיל"""
        rules_filter = RulesFilter()
        
        # בדיקה שעוברת
        is_valid, reason = rules_filter.is_valid_match(
            sample_candidate_female, sample_candidate_male, sample_preferences
        )
        assert is_valid
        
        # בדיקה שלא עוברת - גיל גבוה מדי
        old_candidate = sample_candidate_female.copy()
        old_candidate.age = 35
        
        is_valid, reason = rules_filter.is_valid_match(
            old_candidate, sample_candidate_male, sample_preferences
        )
        assert not is_valid
        assert "גיל" in reason
    
    def test_community_constraint(self, sample_candidate_male, sample_candidate_female, sample_preferences):
        """בדיקת תנאי קהילה"""
        rules_filter = RulesFilter()
        
        # בדיקה שעוברת
        is_valid, reason = rules_filter.is_valid_match(
            sample_candidate_female, sample_candidate_male, sample_preferences
        )
        assert is_valid
        
        # בדיקה שלא עוברת - קהילה לא מתאימה
        different_community_candidate = sample_candidate_female.copy()
        different_community_candidate.community = Community.HASIDIC
        
        is_valid, reason = rules_filter.is_valid_match(
            different_community_candidate, sample_candidate_male, sample_preferences
        )
        assert not is_valid
        assert "קהילה" in reason
    
    def test_smoking_constraint(self, sample_candidate_male, sample_candidate_female, sample_preferences):
        """בדיקת תנאי עישון"""
        rules_filter = RulesFilter()
        
        # בדיקה שעוברת
        is_valid, reason = rules_filter.is_valid_match(
            sample_candidate_female, sample_candidate_male, sample_preferences
        )
        assert is_valid
        
        # בדיקה שלא עוברת - עישון
        smoking_candidate = sample_candidate_female.copy()
        smoking_candidate.smoking = True
        
        is_valid, reason = rules_filter.is_valid_match(
            smoking_candidate, sample_candidate_male, sample_preferences
        )
        assert not is_valid
        assert "עישון" in reason

class TestEmbeddings:
    """בדיקות אמבדינגים"""
    
    def test_hebrew_text_processing(self):
        """בדיקת עיבוד טקסט עברי"""
        embeddings = HebrewEmbeddings()
        
        text = "אני אדם דתי ומשפחתי"
        processed = embeddings.preprocess_text(text)
        
        assert processed == text
        assert isinstance(processed, str)
    
    def test_embedding_generation(self):
        """בדיקת יצירת אמבדינגים"""
        embeddings = HebrewEmbeddings()
        
        text = "אני מחפש בת זוג דתייה ומשפחתית"
        embedding = embeddings.get_embedding(text)
        
        assert embedding is not None
        assert len(embedding) > 0
        assert isinstance(embedding[0], (int, float))
    
    def test_similarity_calculation(self):
        """בדיקת חישוב דמיון"""
        embeddings = HebrewEmbeddings()
        
        text1 = "אני אוהב ללמוד תורה"
        text2 = "אני אוהב לימוד תורה"
        text3 = "אני אוהב ספורט"
        
        # טקסטים דומים
        similarity_high = embeddings.calculate_similarity(text1, text2)
        # טקסטים שונים
        similarity_low = embeddings.calculate_similarity(text1, text3)
        
        assert 0 <= similarity_high <= 1
        assert 0 <= similarity_low <= 1
        assert similarity_high > similarity_low

class TestScoring:
    """בדיקות חישוב ציונים"""
    
    def test_match_score_calculation(self, sample_candidate_male, sample_candidate_female, sample_preferences):
        """בדיקת חישוב ציון התאמה"""
        scorer = MatchingScorer()
        
        # יצירת העדפות למועמדת
        female_preferences = Preferences(
            candidate_id=sample_candidate_female.id,
            must_have=HardConstraints(
                min_age=25,
                max_age=35,
                smoking=False
            ),
            nice_to_have=NiceToHave(),
            free_text="מחפשת בן זוג דתי ומשפחתי"
        )
        
        score = scorer.calculate_match_score(
            sample_candidate_male, sample_candidate_female,
            sample_preferences, female_preferences
        )
        
        assert 0 <= score.total_score <= 1
        assert 0 <= score.semantic_similarity <= 1
        assert 0 <= score.religious_compatibility <= 1
        assert 0 <= score.age_compatibility <= 1
        assert 0 <= score.location_compatibility <= 1
        assert score.explanation is not None
    
    def test_religious_compatibility(self, sample_candidate_male, sample_candidate_female):
        """בדיקת תאימות דתית"""
        scorer = MatchingScorer()
        
        # אותה קהילה ורמת דתיות
        compatibility = scorer.calculate_religious_compatibility(
            sample_candidate_male, sample_candidate_female
        )
        
        assert 0 <= compatibility <= 1
        assert compatibility > 0.5  # אמור להיות גבוה כי אותה קהילה ורמת דתיות
    
    def test_age_compatibility(self, sample_candidate_male, sample_candidate_female):
        """בדיקת תאימות גיל"""
        scorer = MatchingScorer()
        
        compatibility = scorer.calculate_age_compatibility(
            sample_candidate_male, sample_candidate_female
        )
        
        assert 0 <= compatibility <= 1
        
        # הפרש של 3 שנים אמור לתת ציון גבוה
        age_diff = abs(sample_candidate_male.age - sample_candidate_female.age)
        if age_diff <= 5:
            assert compatibility >= 0.6

class TestIntegration:
    """בדיקות אינטגרציה"""
    
    def test_full_matching_pipeline(self, sample_candidate_male, sample_candidate_female, sample_preferences):
        """בדיקת צינור התאמות מלא"""
        # הוספת מועמדים
        client.post("/candidates", json=sample_candidate_male.dict(), headers=HEADERS)
        client.post("/candidates", json=sample_candidate_female.dict(), headers=HEADERS)
        
        # הוספת העדפות
        client.post("/preferences", json=sample_preferences.dict(), headers=HEADERS)
        
        female_preferences = Preferences(
            candidate_id=sample_candidate_female.id,
            must_have=HardConstraints(min_age=25, max_age=35),
            nice_to_have=NiceToHave(),
            free_text="מחפשת בן זוג דתי"
        )
        client.post("/preferences", json=female_preferences.dict(), headers=HEADERS)
        
        # חיפוש התאמות
        search_request = {
            "candidate_id": sample_candidate_male.id,
            "max_results": 10,
            "min_score": 0.0
        }
        
        response = client.post("/match/search", json=search_request, headers=HEADERS)
        assert response.status_code == 200
        
        data = response.json()
        assert "matches" in data
        assert "total_found" in data
        assert "search_time_ms" in data
        
        # אמור למצוא לפחות התאמה אחת
        if data["total_found"] > 0:
            match = data["matches"][0]
            assert "candidate_b_id" in match
            assert "total_score" in match
            assert "explanation" in match

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
