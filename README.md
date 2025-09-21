# מערכת שידוכים היברידית מתקדמת 🤝

מערכת שידוכים מתקדמת המשלבת בינה מלאכותית, כללים קשיחים, אמבדינגים סמנטיים בעברית ולמידת דירוג לחיפוש התאמות מיטביות במגזר החרדי והדתי.

## ✨ תכונות עיקריות

### 🔍 מערכת התאמות היברידית
- **כללים קשיחים**: סינון לפי תנאי סף מחייבים (גיל, מיקום, דתיות, קהילה)
- **אמבדינגים סמנטיים**: ניתוח טקסט חופשי בעברית לחיפוש דמיון בערכים ואישיות
- **למידת דירוג**: שיפור התאמות על בסיס פידבק אמיתי
- **התאמות יציבות**: אלגוריתם Gale-Shapley להתאמות הדדיות

### 🎯 מותאם למגזר החרדי והדתי
- תמיכה מלאה בעברית ובערכים תרבותיים
- רגישות לנושאים דתיים ומסורתיים
- שקיפות בתהליך ההתאמה
- הגנה על פרטיות ומידע רגיש

### 🚀 טכנולוגיות מתקדמות
- **FastAPI** - API מהיר ומודרני
- **Sentence Transformers** - אמבדינגים רב-לשוניים
- **CatBoost/XGBoost** - למידת דירוג מתקדמת
- **Docker** - פריסה קלה ויציבה

## 🏗️ ארכיטקטורה

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   כללים קשיחים   │    │ אמבדינגים סמנטיים │    │   למידת דירוג    │
│   (Rules Filter) │    │   (Embeddings)   │    │    (Ranker)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────┐
                    │      מנוע ציונים        │
                    │    (Scoring Engine)    │
                    └─────────────┬───────────┘
                                 │
                    ┌─────────────▼───────────┐
                    │    התאמות יציבות       │
                    │  (Stable Matching)     │
                    └─────────────────────────┘
```

## 📦 התקנה מהירה

### דרישות מערכת
- Python 3.10+
- Docker (אופציונלי)
- 4GB RAM מינימום
- 2GB שטח דיסק

### התקנה מקומית

```bash
# שכפול הפרויקט
git clone https://github.com/scubapro711/matchmaking-he.git
cd matchmaking-he

# התקנת תלויות
make setup

# יצירת נתוני דוגמה
make dev
```

### התקנה עם Docker

```bash
# בניית התמונה
make docker-build

# הרצת המערכת
make docker-run
```

## 🚀 שימוש מהיר

### הפעלת השרת

```bash
# הפעלה מקומית
make serve

# או עם Docker
docker-compose up
```

השרת יהיה זמין בכתובת: `http://localhost:8000`

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### דוגמאות שימוש

#### הוספת מועמד חדש

```python
import requests

candidate_data = {
    "id": "candidate_001",
    "gender": "M",
    "age": 28,
    "marital_status": "single",
    "community": "lithuanian",
    "religiosity_level": "strict",
    "location": "ירושלים",
    "education": "university",
    "occupation": "מהנדס תוכנה",
    "description_text": "אני אדם דתי ומשפחתי שאוהב ללמוד ולחקור...",
    "languages": ["hebrew", "english"],
    "smoking": false
}

response = requests.post(
    "http://localhost:8000/candidates",
    json=candidate_data,
    headers={"Authorization": "Bearer demo_token_123"}
)
```

#### חיפוש התאמות

```python
search_request = {
    "candidate_id": "candidate_001",
    "max_results": 10,
    "min_score": 0.5,
    "include_explanation": true
}

response = requests.post(
    "http://localhost:8000/match/search",
    json=search_request,
    headers={"Authorization": "Bearer demo_token_123"}
)

matches = response.json()["matches"]
for match in matches:
    print(f"מועמד: {match['candidate_b_id']}")
    print(f"ציון: {match['total_score']:.3f}")
    print(f"הסבר: {match['explanation']}")
    print("---")
```

## 📊 מבנה הנתונים

### מועמד (Candidate)
```json
{
  "id": "candidate_001",
  "gender": "M",
  "age": 28,
  "marital_status": "single",
  "community": "lithuanian",
  "religiosity_level": "strict",
  "location": "ירושלים",
  "education": "university",
  "occupation": "מהנדס תוכנה",
  "description_text": "תיאור אישי...",
  "languages": ["hebrew", "english"],
  "smoking": false
}
```

### העדפות (Preferences)
```json
{
  "candidate_id": "candidate_001",
  "must_have": {
    "min_age": 22,
    "max_age": 32,
    "max_distance_km": 50,
    "required_communities": ["lithuanian", "sephardic"]
  },
  "nice_to_have": {
    "preferred_education": ["university", "advanced_degree"],
    "preferred_location": "ירושלים"
  },
  "free_text": "מחפש בת זוג דתייה ומשפחתית..."
}
```

## 🧮 אלגוריתם הציון

הציון הכולל מחושב לפי הנוסחה:

```
ציון_כולל = 0.55 × דמיון_סמנטי
           + 0.20 × תאימות_דתית
           + 0.10 × התאמת_גיל  
           + 0.10 × קרבה_גיאוגרפית
           + 0.05 × גורמים_נוספים
```

### רכיבי הציון

1. **דמיון סמנטי (55%)**: ניתוח טקסט חופשי באמצעות אמבדינגים
2. **תאימות דתית (20%)**: קהילה ורמת דתיות
3. **התאמת גיל (10%)**: הפרש גילאים
4. **קרבה גיאוגרפית (10%)**: מרחק פיזי
5. **גורמים נוספים (5%)**: השכלה, שפות, עישון

## 🤖 למידת דירוג

המערכת משתפרת עם הזמן באמצעות פידבק:

### סוגי פידבק
- ✅ **התאמה מוצלחת** (ציון: 5)
- 📅 **נקבעה פגישה** (ציון: 4)  
- 📞 **נוצר קשר** (ציון: 3)
- 📧 **נשלחה הצעה** (ציון: 2)
- ❌ **נדחה** (ציון: 0)

### אימון המודל

```bash
# אימון מודל הדירוג
make rank

# או דרך API
curl -X POST "http://localhost:8000/ranker/train" \
  -H "Authorization: Bearer demo_token_123"
```

## 🔒 אבטחה ופרטיות

### הגנת מידע
- הצפנת נתונים רגישים
- הרשאות מבוססות טוקן
- לוגים ללא מידע אישי
- מזעור איסוף נתונים

### תאימות תרבותית
- שפה נקייה ומכבדת
- התאמה לערכים דתיים
- שקיפות בתהליכים
- מנגנון ערעור ותיקון

## 📈 מדדי ביצועים

### מדדי איכות
- **Precision@K**: דיוק ההמלצות
- **nDCG@K**: איכות הדירוג
- **שיעור התאמות מוצלחות**: אחוז ההתאמות שהובילו לפגישה/שידוך

### מדדי מערכת
- זמן תגובה ממוצע: < 500ms
- זמינות: 99.9%
- תפוקה: 1000+ בקשות/דקה

## 🛠️ פיתוח

### הרצת בדיקות

```bash
# בדיקות יחידה
make test

# בדיקות כיסוי
pytest --cov=src tests/
```

### מבנה הפרויקט

```
matchmaking-he/
├── src/                    # קוד המקור
│   ├── data_schemas.py     # סכימות נתונים
│   ├── rules_filter.py     # מסנני כללים
│   ├── embeddings.py       # אמבדינגים סמנטיים
│   ├── scoring.py          # חישוב ציונים
│   ├── ranker.py          # למידת דירוג
│   ├── stable_matching.py # התאמות יציבות
│   ├── api.py             # FastAPI
│   └── data_generator.py  # מחולל נתוני דוגמה
├── data/                  # נתונים
├── models/                # מודלים מאומנים
├── tests/                 # בדיקות
├── docker/                # קבצי Docker
├── notebooks/             # Jupyter notebooks
├── requirements.txt       # תלויות Python
├── Makefile              # פקודות אוטומציה
└── README.md             # תיעוד זה
```

## 🤝 תרומה לפרויקט

אנו מזמינים תרומות! אנא קראו את [CONTRIBUTING.md](CONTRIBUTING.md) לפרטים.

### תהליך התרומה
1. Fork הפרויקט
2. יצירת branch לפיצ'ר (`git checkout -b feature/amazing-feature`)
3. Commit השינויים (`git commit -m 'Add amazing feature'`)
4. Push ל-branch (`git push origin feature/amazing-feature`)
5. פתיחת Pull Request

## 📄 רישיון

פרויקט זה מופץ תחת רישיון MIT. ראו [LICENSE](LICENSE) לפרטים.

## 📞 יצירת קשר

- **GitHub Issues**: לבאגים ובקשות פיצ'רים
- **Email**: [your-email@example.com]
- **Documentation**: [Wiki](https://github.com/scubapro711/matchmaking-he/wiki)

## 🙏 תודות

- **Sentence Transformers** - למודלי האמבדינגים הרב-לשוניים
- **CatBoost** - למודל למידת הדירוג
- **FastAPI** - לפריימוורק ה-API המהיר
- **הקהילה הדתית** - על המשוב והכוונה התרבותית

---

**מערכת שידוכים היברידית** - מחברת בין מסורת לחדשנות טכנולוגית 💙
