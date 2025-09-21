"""
מודול חישוב ציונים למערכת השידוכים
"""
import numpy as np
from typing import List, Dict, Tuple
from src.data_schemas import Candidate, Preferences, MatchScore, Community, ReligiosityLevel
from src.embeddings import HebrewEmbeddings
from src.rules_filter import RulesFilter
import logging

logger = logging.getLogger(__name__)

class MatchingScorer:
    """מחלקה לחישוב ציוני התאמה בין מועמדים"""
    
    def __init__(self):
        self.embeddings = HebrewEmbeddings()
        self.rules_filter = RulesFilter()
        
        # משקלות ברירת מחדל
        self.weights = {
            'semantic_similarity': 0.55,
            'religious_compatibility': 0.20,
            'age_compatibility': 0.10,
            'location_compatibility': 0.10,
            'other_factors': 0.05
        }
        
        # מטריצות תאימות
        self._build_compatibility_matrices()
    
    def _build_compatibility_matrices(self):
        """בניית מטריצות תאימות לקהילות ורמות דתיות"""
        
        # מטריצת תאימות קהילות (0-1)
        self.community_compatibility = {
            Community.LITHUANIAN: {
                Community.LITHUANIAN: 1.0,
                Community.HASIDIC: 0.6,
                Community.SEPHARDIC: 0.7,
                Community.MODERN_ORTHODOX: 0.4,
                Community.NATIONAL_RELIGIOUS: 0.5
            },
            Community.HASIDIC: {
                Community.LITHUANIAN: 0.6,
                Community.HASIDIC: 1.0,
                Community.SEPHARDIC: 0.5,
                Community.MODERN_ORTHODOX: 0.3,
                Community.NATIONAL_RELIGIOUS: 0.4
            },
            Community.SEPHARDIC: {
                Community.LITHUANIAN: 0.7,
                Community.HASIDIC: 0.5,
                Community.SEPHARDIC: 1.0,
                Community.MODERN_ORTHODOX: 0.6,
                Community.NATIONAL_RELIGIOUS: 0.7
            },
            Community.MODERN_ORTHODOX: {
                Community.LITHUANIAN: 0.4,
                Community.HASIDIC: 0.3,
                Community.SEPHARDIC: 0.6,
                Community.MODERN_ORTHODOX: 1.0,
                Community.NATIONAL_RELIGIOUS: 0.8
            },
            Community.NATIONAL_RELIGIOUS: {
                Community.LITHUANIAN: 0.5,
                Community.HASIDIC: 0.4,
                Community.SEPHARDIC: 0.7,
                Community.MODERN_ORTHODOX: 0.8,
                Community.NATIONAL_RELIGIOUS: 1.0
            }
        }
        
        # מטריצת תאימות רמות דתיות
        self.religiosity_compatibility = {
            ReligiosityLevel.VERY_STRICT: {
                ReligiosityLevel.VERY_STRICT: 1.0,
                ReligiosityLevel.STRICT: 0.8,
                ReligiosityLevel.MODERATE: 0.4,
                ReligiosityLevel.FLEXIBLE: 0.2
            },
            ReligiosityLevel.STRICT: {
                ReligiosityLevel.VERY_STRICT: 0.8,
                ReligiosityLevel.STRICT: 1.0,
                ReligiosityLevel.MODERATE: 0.7,
                ReligiosityLevel.FLEXIBLE: 0.4
            },
            ReligiosityLevel.MODERATE: {
                ReligiosityLevel.VERY_STRICT: 0.4,
                ReligiosityLevel.STRICT: 0.7,
                ReligiosityLevel.MODERATE: 1.0,
                ReligiosityLevel.FLEXIBLE: 0.8
            },
            ReligiosityLevel.FLEXIBLE: {
                ReligiosityLevel.VERY_STRICT: 0.2,
                ReligiosityLevel.STRICT: 0.4,
                ReligiosityLevel.MODERATE: 0.8,
                ReligiosityLevel.FLEXIBLE: 1.0
            }
        }
    
    def calculate_semantic_similarity(self, candidate_a: Candidate, candidate_b: Candidate,
                                    preferences_a: Preferences, preferences_b: Preferences) -> float:
        """חישוב דמיון סמנטי בין שני מועמדים"""
        
        # שילוב תיאור אישי והעדפות
        text_a = f"{candidate_a.description_text} {preferences_a.free_text}"
        text_b = f"{candidate_b.description_text} {preferences_b.free_text}"
        
        similarity = self.embeddings.calculate_similarity(text_a, text_b)
        
        logger.debug(f"Semantic similarity between {candidate_a.id} and {candidate_b.id}: {similarity:.3f}")
        return similarity
    
    def calculate_religious_compatibility(self, candidate_a: Candidate, candidate_b: Candidate) -> float:
        """חישוב תאימות דתית"""
        
        # תאימות קהילה
        community_score = self.community_compatibility.get(
            candidate_a.community, {}
        ).get(candidate_b.community, 0.5)
        
        # תאימות רמת דתיות
        religiosity_score = self.religiosity_compatibility.get(
            candidate_a.religiosity_level, {}
        ).get(candidate_b.religiosity_level, 0.5)
        
        # ממוצע משוקלל
        religious_compatibility = (community_score * 0.6) + (religiosity_score * 0.4)
        
        logger.debug(f"Religious compatibility: community={community_score:.3f}, "
                    f"religiosity={religiosity_score:.3f}, total={religious_compatibility:.3f}")
        
        return religious_compatibility
    
    def calculate_age_compatibility(self, candidate_a: Candidate, candidate_b: Candidate) -> float:
        """חישוב תאימות גיל"""
        
        age_diff = abs(candidate_a.age - candidate_b.age)
        
        # פונקציה יורדת - ככל שההפרש גדול יותר, הציון נמוך יותר
        if age_diff <= 2:
            score = 1.0
        elif age_diff <= 5:
            score = 0.8
        elif age_diff <= 8:
            score = 0.6
        elif age_diff <= 12:
            score = 0.4
        else:
            score = 0.2
        
        logger.debug(f"Age compatibility: diff={age_diff}, score={score:.3f}")
        return score
    
    def calculate_location_compatibility(self, candidate_a: Candidate, candidate_b: Candidate) -> float:
        """חישוב תאימות מיקום"""
        
        try:
            distance = self.rules_filter.calculate_distance(candidate_a.location, candidate_b.location)
            
            # פונקציה יורדת לפי מרחק
            if distance <= 10:
                score = 1.0
            elif distance <= 25:
                score = 0.8
            elif distance <= 50:
                score = 0.6
            elif distance <= 100:
                score = 0.4
            else:
                score = 0.2
            
            logger.debug(f"Location compatibility: distance={distance:.1f}km, score={score:.3f}")
            return score
            
        except Exception as e:
            logger.warning(f"Failed to calculate location compatibility: {e}")
            return 0.5
    
    def calculate_other_factors(self, candidate_a: Candidate, candidate_b: Candidate,
                              preferences_a: Preferences, preferences_b: Preferences) -> float:
        """חישוב גורמים נוספים"""
        
        factors_score = 0.0
        factor_count = 0
        
        # תאימות השכלה
        education_compatibility = 1.0 if candidate_a.education == candidate_b.education else 0.7
        factors_score += education_compatibility
        factor_count += 1
        
        # תאימות שפות
        common_languages = set(candidate_a.languages) & set(candidate_b.languages)
        language_score = min(1.0, len(common_languages) / max(len(candidate_a.languages), 1))
        factors_score += language_score
        factor_count += 1
        
        # תאימות עישון
        smoking_score = 1.0 if candidate_a.smoking == candidate_b.smoking else 0.3
        factors_score += smoking_score
        factor_count += 1
        
        # ממוצע
        final_score = factors_score / factor_count if factor_count > 0 else 0.5
        
        logger.debug(f"Other factors: education={education_compatibility:.3f}, "
                    f"language={language_score:.3f}, smoking={smoking_score:.3f}, "
                    f"total={final_score:.3f}")
        
        return final_score
    
    def calculate_match_score(self, candidate_a: Candidate, candidate_b: Candidate,
                            preferences_a: Preferences, preferences_b: Preferences) -> MatchScore:
        """חישוב ציון התאמה מקיף"""
        
        # חישוב כל הרכיבים
        semantic_similarity = self.calculate_semantic_similarity(
            candidate_a, candidate_b, preferences_a, preferences_b
        )
        
        religious_compatibility = self.calculate_religious_compatibility(
            candidate_a, candidate_b
        )
        
        age_compatibility = self.calculate_age_compatibility(
            candidate_a, candidate_b
        )
        
        location_compatibility = self.calculate_location_compatibility(
            candidate_a, candidate_b
        )
        
        other_factors = self.calculate_other_factors(
            candidate_a, candidate_b, preferences_a, preferences_b
        )
        
        # חישוב ציון כולל
        total_score = (
            semantic_similarity * self.weights['semantic_similarity'] +
            religious_compatibility * self.weights['religious_compatibility'] +
            age_compatibility * self.weights['age_compatibility'] +
            location_compatibility * self.weights['location_compatibility'] +
            other_factors * self.weights['other_factors']
        )
        
        # יצירת הסבר
        explanation = self._generate_explanation(
            semantic_similarity, religious_compatibility, age_compatibility,
            location_compatibility, other_factors, total_score
        )
        
        return MatchScore(
            candidate_a_id=candidate_a.id,
            candidate_b_id=candidate_b.id,
            total_score=total_score,
            semantic_similarity=semantic_similarity,
            religious_compatibility=religious_compatibility,
            age_compatibility=age_compatibility,
            location_compatibility=location_compatibility,
            other_factors=other_factors,
            explanation=explanation
        )
    
    def _generate_explanation(self, semantic: float, religious: float, age: float,
                            location: float, other: float, total: float) -> str:
        """יצירת הסבר על הציון"""
        
        explanations = []
        
        if semantic >= 0.8:
            explanations.append("תאימות גבוהה בערכים ובאישיות")
        elif semantic >= 0.6:
            explanations.append("תאימות טובה בערכים")
        else:
            explanations.append("תאימות בסיסית בערכים")
        
        if religious >= 0.8:
            explanations.append("תאימות מעולה ברקע הדתי")
        elif religious >= 0.6:
            explanations.append("תאימות טובה ברקע הדתי")
        
        if age >= 0.8:
            explanations.append("הפרש גיל מתאים")
        elif age < 0.5:
            explanations.append("הפרש גיל גדול יחסית")
        
        if location >= 0.8:
            explanations.append("קרבה גיאוגרפית טובה")
        elif location < 0.5:
            explanations.append("מרחק גיאוגרפי גדול")
        
        if total >= 0.8:
            overall = "התאמה מעולה"
        elif total >= 0.6:
            overall = "התאמה טובה"
        elif total >= 0.4:
            overall = "התאמה בינונית"
        else:
            overall = "התאמה חלשה"
        
        return f"{overall}: {', '.join(explanations)}"
    
    def update_weights(self, new_weights: Dict[str, float]):
        """עדכון משקלות הציון"""
        
        # וידוא שסכום המשקלות הוא 1
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Sum of weights must be 1.0, got {total_weight}")
        
        self.weights.update(new_weights)
        logger.info(f"Updated scoring weights: {self.weights}")
    
    def batch_score_candidates(self, requester: Candidate, candidates: List[Candidate],
                             requester_preferences: Preferences,
                             candidates_preferences: Dict[str, Preferences]) -> List[MatchScore]:
        """חישוב ציונים לרשימת מועמדים"""
        
        scores = []
        
        for candidate in candidates:
            if candidate.id == requester.id:
                continue
            
            candidate_preferences = candidates_preferences.get(candidate.id)
            if not candidate_preferences:
                logger.warning(f"No preferences found for candidate {candidate.id}")
                continue
            
            try:
                score = self.calculate_match_score(
                    requester, candidate, requester_preferences, candidate_preferences
                )
                scores.append(score)
                
            except Exception as e:
                logger.error(f"Failed to calculate score for candidate {candidate.id}: {e}")
                continue
        
        # מיון לפי ציון
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        logger.info(f"Calculated scores for {len(scores)} candidates")
        return scores
