"""
מחולל נתוני דוגמה למערכת השידוכים
"""
import json
import random
from datetime import datetime, timedelta
from typing import List
from src.data_schemas import (
    Candidate, Preferences, Feedback, Gender, MaritalStatus, 
    Community, ReligiosityLevel, EducationLevel, HardConstraints, 
    NiceToHave, FeedbackStatus
)

class DataGenerator:
    """מחלקה ליצירת נתוני דוגמה"""
    
    def __init__(self):
        self.hebrew_names_male = [
            "אברהם", "יצחק", "יעקב", "משה", "אהרן", "דוד", "שלמה", "יוסף", "בנימין", "לוי",
            "יהודה", "ראובן", "שמעון", "זבולון", "יששכר", "דן", "נפתלי", "גד", "אשר", "אפרים"
        ]
        
        self.hebrew_names_female = [
            "שרה", "רבקה", "רחל", "לאה", "מרים", "דבורה", "יהודית", "אסתר", "רות", "נעמי",
            "חנה", "תמר", "אביגיל", "בת שבע", "מיכל", "חולדה", "גומר", "עטרה", "שושנה", "יעל"
        ]
        
        self.cities = [
            "ירושלים", "תל אביב", "חיפה", "ראשון לציון", "פתח תקווה", "אשדוד", "נתניה", 
            "באר שבע", "בני ברק", "חולון", "רמת גן", "אשקלון", "רחובות", "בת ים", "כפר סבא",
            "הרצליה", "מודיעין", "לוד", "רמלה", "נצרת", "עכו", "נהריה", "טבריה", "צפת",
            "בית שמש", "אלעד", "עמנואל", "ביתר עילית", "קרית ספר"
        ]
        
        self.occupations = [
            "מהנדס", "רופא", "עורך דין", "מורה", "אדריכל", "חשב", "מתכנת", "פסיכולוג",
            "רוקח", "פיזיותרפיסט", "מעצב גרפי", "יועץ עסקי", "מנהל פרויקטים", "מחקר ופיתוח",
            "עובד סוציאלי", "מטפל בילדים", "מנהל כספים", "יועץ השקעות", "מנהל מכירות",
            "מומחה שיווק", "מנהל משאבי אנוש", "יועץ ארגוני", "מנהל תפעול", "מומחה IT"
        ]
        
        self.description_templates = [
            "אני אדם {personality} שאוהב {hobby} ומחפש בן/בת זוג {partner_trait}. חשוב לי {value} ואני מעוניין ב{relationship_goal}.",
            "בן/בת {age} {education_desc}, {personality} ו{trait}. אוהב {hobby} ומחפש מישהו {partner_trait} לבניית {relationship_goal}.",
            "אני {profession} {personality} שמחפש בן/בת זוג {partner_trait}. חשוב לי {value} ואני רוצה לבנות {relationship_goal}.",
            "בן/בת {age} מ{city}, {personality} ואוהב {hobby}. מחפש מישהו {partner_trait} שחולק איתי את הערכים של {value}."
        ]
        
        self.personality_traits = [
            "חם ומשפחתי", "אינטלקטואל וסקרן", "רגוש ואמפתי", "עליז ואופטימי", 
            "רציני ואמין", "יצירתי ומקורי", "שקט ומתבונן", "חברותי ופתוח"
        ]
        
        self.hobbies = [
            "קריאה ולמידה", "ספורט ופעילות גופנית", "מוזיקה ותרבות", "בישול ואפייה",
            "טיולים בטבע", "צילום ואמנות", "כתיבה ויצירה", "משחקי חשיבה", 
            "פעילות קהילתית", "עבודת יד ויצירה"
        ]
        
        self.partner_traits = [
            "דתי ומשפחתי", "אינטלקטואל ומשכיל", "חם ואוהב", "יציב ואמין",
            "עם חוש הומור", "רגיש ומבין", "עם ערכים חזקים", "פתוח ותומך"
        ]
        
        self.values = [
            "משפחה ומסורת", "לימוד תורה וערכים", "כנות ואמת", "צמיחה אישית",
            "עזרה לזולת", "יושרה ומוסר", "שלום בית", "חינוך הילדים"
        ]
        
        self.relationship_goals = [
            "בית נאמן בישראל", "משפחה גדולה ומאושרת", "קשר עמוק ומשמעותי",
            "שותפות לחיים", "בית מלא אהבה", "משפחה דתית ומסורתית"
        ]
    
    def generate_description(self, candidate: Candidate) -> str:
        """יצירת תיאור אישי"""
        template = random.choice(self.description_templates)
        
        # התאמת תבנית למגדר
        if candidate.gender == Gender.MALE:
            template = template.replace("בן/בת", "בן").replace("מישהו", "מישהי")
        else:
            template = template.replace("בן/בת", "בת").replace("מישהו", "מישהו")
        
        return template.format(
            personality=random.choice(self.personality_traits),
            hobby=random.choice(self.hobbies),
            partner_trait=random.choice(self.partner_traits),
            value=random.choice(self.values),
            relationship_goal=random.choice(self.relationship_goals),
            age=candidate.age,
            education_desc=self.get_education_description(candidate.education),
            trait=random.choice(self.personality_traits),
            profession=random.choice(self.occupations),
            city=candidate.location
        )
    
    def get_education_description(self, education: EducationLevel) -> str:
        """תיאור השכלה"""
        descriptions = {
            EducationLevel.HIGH_SCHOOL: "בוגר תיכון",
            EducationLevel.SEMINARY: "בוגר סמינר",
            EducationLevel.YESHIVA: "בוגר ישיבה",
            EducationLevel.COLLEGE: "בוגר מכללה",
            EducationLevel.UNIVERSITY: "בוגר אוניברסיטה",
            EducationLevel.ADVANCED_DEGREE: "בעל תואר מתקדם"
        }
        return descriptions.get(education, "בעל השכלה")
    
    def generate_candidate(self, candidate_id: str) -> Candidate:
        """יצירת מועמד"""
        gender = random.choice(list(Gender))
        age = random.randint(20, 45)
        
        # בחירת שם לפי מגדר
        if gender == Gender.MALE:
            name = random.choice(self.hebrew_names_male)
        else:
            name = random.choice(self.hebrew_names_female)
        
        candidate = Candidate(
            id=candidate_id,
            gender=gender,
            age=age,
            marital_status=random.choice(list(MaritalStatus)),
            community=random.choice(list(Community)),
            religiosity_level=random.choice(list(ReligiosityLevel)),
            location=random.choice(self.cities),
            education=random.choice(list(EducationLevel)),
            occupation=random.choice(self.occupations),
            description_text="",  # נמלא אחר כך
            languages=["hebrew"] + (["english"] if random.random() > 0.3 else []),
            smoking=random.random() < 0.1,  # 10% מעשנים
            full_name=name
        )
        
        # יצירת תיאור אישי
        candidate.description_text = self.generate_description(candidate)
        
        return candidate
    
    def generate_preferences(self, candidate: Candidate) -> Preferences:
        """יצירת העדפות למועמד"""
        
        # תנאי סף
        age_range = random.randint(5, 15)
        must_have = HardConstraints(
            min_age=max(18, candidate.age - age_range),
            max_age=min(120, candidate.age + age_range),
            max_distance_km=random.choice([50, 100, 200, None]),
            smoking=None if random.random() > 0.7 else (not candidate.smoking),
            required_communities=[candidate.community] if random.random() > 0.4 else None,
            required_religiosity=[candidate.religiosity_level] if random.random() > 0.5 else None,
            required_languages=["hebrew"] if random.random() > 0.8 else None
        )
        
        # העדפות רכות
        nice_to_have = NiceToHave(
            preferred_education=[candidate.education] if random.random() > 0.6 else None,
            preferred_occupation=[candidate.occupation] if random.random() > 0.8 else None,
            preferred_location=candidate.location if random.random() > 0.7 else None
        )
        
        # טקסט חופשי
        free_text_options = [
            f"מחפש בן/בת זוג {random.choice(self.partner_traits)} שחולק איתי את הערכים של {random.choice(self.values)}",
            f"חשוב לי {random.choice(self.values)} ואני רוצה לבנות {random.choice(self.relationship_goals)}",
            f"אני {random.choice(self.personality_traits)} ומחפש מישהו שאוהב {random.choice(self.hobbies)}",
            f"מעוניין במישהו {random.choice(self.partner_traits)} לבניית {random.choice(self.relationship_goals)}"
        ]
        
        free_text = random.choice(free_text_options)
        if candidate.gender == Gender.MALE:
            free_text = free_text.replace("בן/בת", "בת").replace("מישהו", "מישהי")
        else:
            free_text = free_text.replace("בן/בת", "בן").replace("מישהו", "מישהו")
        
        return Preferences(
            candidate_id=candidate.id,
            must_have=must_have,
            nice_to_have=nice_to_have,
            free_text=free_text
        )
    
    def generate_feedback(self, candidate_a_id: str, candidate_b_id: str) -> Feedback:
        """יצירת פידבק"""
        
        # הסתברויות שונות לסטטוסים שונים
        status_weights = {
            FeedbackStatus.SENT: 0.3,
            FeedbackStatus.CONTACT_MADE: 0.25,
            FeedbackStatus.MEETING_ARRANGED: 0.2,
            FeedbackStatus.MATCHED: 0.05,
            FeedbackStatus.REJECTED: 0.15,
            FeedbackStatus.NO_RESPONSE: 0.05
        }
        
        status = random.choices(
            list(status_weights.keys()),
            weights=list(status_weights.values())
        )[0]
        
        # סיבות לפי סטטוס
        reasons = {
            FeedbackStatus.REJECTED: [
                "לא מתאים מבחינת גיל", "הבדלים בערכים", "לא מתאים מבחינת מיקום",
                "הבדלים ברמת דתיות", "לא היה חיבור אישי", "הבדלים בציפיות"
            ],
            FeedbackStatus.NO_RESPONSE: ["לא הגיב", "לא זמין כרגע", "החליט לא להמשיך"],
            FeedbackStatus.MATCHED: ["התאמה מושלמת!", "חתונה בקרוב", "מאושרים מאוד"]
        }
        
        reason = None
        if status in reasons:
            reason = random.choice(reasons[status])
        
        rating = None
        if status in [FeedbackStatus.CONTACT_MADE, FeedbackStatus.MEETING_ARRANGED, FeedbackStatus.MATCHED]:
            rating = random.randint(3, 5)
        elif status == FeedbackStatus.REJECTED:
            rating = random.randint(1, 3)
        
        return Feedback(
            candidate_a_id=candidate_a_id,
            candidate_b_id=candidate_b_id,
            status=status,
            reason=reason,
            rating=rating,
            timestamp=datetime.now() - timedelta(days=random.randint(0, 365))
        )
    
    def generate_sample_data(self, num_candidates: int = 50, num_feedback: int = 100):
        """יצירת מערך נתונים לדוגמה"""
        
        print(f"Generating {num_candidates} candidates...")
        
        # יצירת מועמדים
        candidates = []
        preferences = []
        
        for i in range(num_candidates):
            candidate_id = f"candidate_{i+1:03d}"
            candidate = self.generate_candidate(candidate_id)
            preference = self.generate_preferences(candidate)
            
            candidates.append(candidate)
            preferences.append(preference)
        
        # יצירת פידבק
        print(f"Generating {num_feedback} feedback entries...")
        feedback_list = []
        
        for _ in range(num_feedback):
            candidate_a = random.choice(candidates)
            candidate_b = random.choice(candidates)
            
            # וידוא שלא מדובר באותו מועמד ושמגדרים שונים
            if (candidate_a.id != candidate_b.id and 
                candidate_a.gender != candidate_b.gender):
                
                feedback = self.generate_feedback(candidate_a.id, candidate_b.id)
                feedback_list.append(feedback)
        
        # שמירה לקבצים
        print("Saving data to files...")
        
        # המרה ל-dict לשמירה
        candidates_data = [candidate.dict() for candidate in candidates]
        preferences_data = [preference.dict() for preference in preferences]
        feedback_data = [feedback.dict() for feedback in feedback_list]
        
        with open("data/candidates.json", "w", encoding="utf-8") as f:
            json.dump(candidates_data, f, ensure_ascii=False, indent=2, default=str)
        
        with open("data/preferences.json", "w", encoding="utf-8") as f:
            json.dump(preferences_data, f, ensure_ascii=False, indent=2, default=str)
        
        with open("data/feedback.json", "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Generated data saved:")
        print(f"  - {len(candidates)} candidates")
        print(f"  - {len(preferences)} preferences")
        print(f"  - {len(feedback_list)} feedback entries")
        
        return candidates, preferences, feedback_list

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Generate sample data for matchmaking system")
    parser.add_argument("--generate-sample", action="store_true", help="Generate sample data")
    parser.add_argument("--candidates", type=int, default=50, help="Number of candidates to generate")
    parser.add_argument("--feedback", type=int, default=100, help="Number of feedback entries to generate")
    
    args = parser.parse_args()
    
    if args.generate_sample:
        # יצירת תיקיית data אם לא קיימת
        os.makedirs("data", exist_ok=True)
        
        generator = DataGenerator()
        generator.generate_sample_data(args.candidates, args.feedback)
