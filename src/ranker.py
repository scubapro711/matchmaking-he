"""
מודול למידת דירוג (Learning-to-Rank) למערכת השידוכים
"""
import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from catboost import CatBoostRanker
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from src.data_schemas import Candidate, Preferences, Feedback, FeedbackStatus
import logging

logger = logging.getLogger(__name__)

class MatchingRanker:
    """מחלקה ללמידת דירוג התאמות"""
    
    def __init__(self, model_path: str = "models/ranker_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_names = []
        self.is_trained = False
        
        # מיפוי סטטוס פידבק לציון
        self.feedback_scores = {
            FeedbackStatus.MATCHED: 5,
            FeedbackStatus.MEETING_ARRANGED: 4,
            FeedbackStatus.CONTACT_MADE: 3,
            FeedbackStatus.SENT: 2,
            FeedbackStatus.NO_RESPONSE: 1,
            FeedbackStatus.REJECTED: 0
        }
        
        self._load_model()
    
    def _load_model(self):
        """טעינת מודל קיים"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.feature_names = model_data['feature_names']
                    self.is_trained = True
                logger.info(f"Loaded trained ranker model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ranker model: {e}")
                self.model = None
                self.is_trained = False
    
    def _save_model(self):
        """שמירת המודל"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Saved ranker model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save ranker model: {e}")
    
    def extract_features(self, candidate_a: Candidate, candidate_b: Candidate,
                        preferences_a: Preferences, preferences_b: Preferences,
                        base_scores: Dict[str, float]) -> Dict[str, float]:
        """חילוץ פיצ'רים לדירוג"""
        
        features = {}
        
        # פיצ'רים מהציונים הבסיסיים
        features.update(base_scores)
        
        # פיצ'רים דמוגרפיים
        features['age_diff'] = abs(candidate_a.age - candidate_b.age)
        features['age_a'] = candidate_a.age
        features['age_b'] = candidate_b.age
        
        # פיצ'רים קטגוריאליים (מקודדים)
        features['same_community'] = 1.0 if candidate_a.community == candidate_b.community else 0.0
        features['same_religiosity'] = 1.0 if candidate_a.religiosity_level == candidate_b.religiosity_level else 0.0
        features['same_education'] = 1.0 if candidate_a.education == candidate_b.education else 0.0
        features['same_smoking'] = 1.0 if candidate_a.smoking == candidate_b.smoking else 0.0
        
        # פיצ'רי שפות
        common_languages = set(candidate_a.languages) & set(candidate_b.languages)
        features['common_languages_count'] = len(common_languages)
        features['language_overlap_ratio'] = len(common_languages) / max(len(candidate_a.languages), 1)
        
        # פיצ'רי טקסט
        features['description_length_a'] = len(candidate_a.description_text.split())
        features['description_length_b'] = len(candidate_b.description_text.split())
        features['preferences_length_a'] = len(preferences_a.free_text.split())
        features['preferences_length_b'] = len(preferences_b.free_text.split())
        
        # פיצ'רי העדפות
        prefs_a = preferences_a.must_have
        prefs_b = preferences_b.must_have
        
        # בדיקת התאמה להעדפות גיל
        if prefs_a.min_age and prefs_a.max_age:
            features['age_in_range_a'] = 1.0 if prefs_a.min_age <= candidate_b.age <= prefs_a.max_age else 0.0
        else:
            features['age_in_range_a'] = 0.5
            
        if prefs_b.min_age and prefs_b.max_age:
            features['age_in_range_b'] = 1.0 if prefs_b.min_age <= candidate_a.age <= prefs_b.max_age else 0.0
        else:
            features['age_in_range_b'] = 0.5
        
        # פיצ'רי קהילה והעדפות
        if prefs_a.required_communities:
            features['community_match_a'] = 1.0 if candidate_b.community in prefs_a.required_communities else 0.0
        else:
            features['community_match_a'] = 0.5
            
        if prefs_b.required_communities:
            features['community_match_b'] = 1.0 if candidate_a.community in prefs_b.required_communities else 0.0
        else:
            features['community_match_b'] = 0.5
        
        return features
    
    def prepare_training_data(self, feedback_data: List[Feedback],
                            candidates_data: Dict[str, Candidate],
                            preferences_data: Dict[str, Preferences],
                            base_scores_data: Dict[Tuple[str, str], Dict[str, float]]) -> Tuple[pd.DataFrame, List[int], List[int]]:
        """הכנת נתונים לאימון"""
        
        training_features = []
        training_labels = []
        group_sizes = []
        
        # קיבוץ פידבק לפי מועמד מבקש
        feedback_by_requester = {}
        for feedback in feedback_data:
            requester_id = feedback.candidate_a_id
            if requester_id not in feedback_by_requester:
                feedback_by_requester[requester_id] = []
            feedback_by_requester[requester_id].append(feedback)
        
        for requester_id, requester_feedback in feedback_by_requester.items():
            if requester_id not in candidates_data or requester_id not in preferences_data:
                continue
            
            requester = candidates_data[requester_id]
            requester_prefs = preferences_data[requester_id]
            
            group_features = []
            group_labels = []
            
            for feedback in requester_feedback:
                candidate_id = feedback.candidate_b_id
                
                if candidate_id not in candidates_data or candidate_id not in preferences_data:
                    continue
                
                candidate = candidates_data[candidate_id]
                candidate_prefs = preferences_data[candidate_id]
                
                # קבלת ציונים בסיסיים
                score_key = (requester_id, candidate_id)
                if score_key not in base_scores_data:
                    continue
                
                base_scores = base_scores_data[score_key]
                
                # חילוץ פיצ'רים
                features = self.extract_features(
                    requester, candidate, requester_prefs, candidate_prefs, base_scores
                )
                
                # המרת פידבק לציון
                label = self.feedback_scores.get(feedback.status, 1)
                
                group_features.append(features)
                group_labels.append(label)
            
            if len(group_features) > 1:  # צריך לפחות 2 דוגמאות לקבוצה
                training_features.extend(group_features)
                training_labels.extend(group_labels)
                group_sizes.append(len(group_features))
        
        if not training_features:
            raise ValueError("No valid training data found")
        
        # המרה ל-DataFrame
        df = pd.DataFrame(training_features)
        self.feature_names = list(df.columns)
        
        logger.info(f"Prepared training data: {len(training_features)} samples, "
                   f"{len(group_sizes)} groups, {len(self.feature_names)} features")
        
        return df, training_labels, group_sizes
    
    def train(self, feedback_data: List[Feedback],
             candidates_data: Dict[str, Candidate],
             preferences_data: Dict[str, Preferences],
             base_scores_data: Dict[Tuple[str, str], Dict[str, float]]):
        """אימון מודל הדירוג"""
        
        logger.info("Starting ranker training...")
        
        # הכנת נתוני אימון
        X, y, group_sizes = self.prepare_training_data(
            feedback_data, candidates_data, preferences_data, base_scores_data
        )
        
        if len(X) < 10:
            logger.warning(f"Very little training data ({len(X)} samples). Model may not be reliable.")
        
        # פיצול לאימון ובדיקה
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # חישוב group_sizes חדש לאחר הפיצול
        train_groups = [len(X_train)]  # פישוט - קבוצה אחת גדולה
        test_groups = [len(X_test)]
        
        # אימון המודל
        self.model = CatBoostRanker(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function='YetiRank',
            verbose=False,
            random_seed=42
        )
        
        try:
            self.model.fit(
                X_train, y_train,
                group_id=train_groups,
                eval_set=(X_test, y_test),
                verbose=False
            )
            
            self.is_trained = True
            
            # הערכת המודל
            y_pred = self.model.predict(X_test)
            ndcg = ndcg_score([y_test], [y_pred])
            
            logger.info(f"Ranker training completed. NDCG score: {ndcg:.4f}")
            
            # שמירת המודל
            self._save_model()
            
        except Exception as e:
            logger.error(f"Failed to train ranker: {e}")
            raise
    
    def predict_score(self, candidate_a: Candidate, candidate_b: Candidate,
                     preferences_a: Preferences, preferences_b: Preferences,
                     base_scores: Dict[str, float]) -> float:
        """חיזוי ציון דירוג למועמד"""
        
        if not self.is_trained:
            logger.warning("Ranker not trained, returning base total score")
            return base_scores.get('total_score', 0.5)
        
        try:
            # חילוץ פיצ'רים
            features = self.extract_features(
                candidate_a, candidate_b, preferences_a, preferences_b, base_scores
            )
            
            # וידוא שכל הפיצ'רים קיימים
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            # חיזוי
            score = self.model.predict([feature_vector])[0]
            
            # נרמול לטווח 0-1
            normalized_score = max(0.0, min(1.0, score / 5.0))
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Failed to predict ranker score: {e}")
            return base_scores.get('total_score', 0.5)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """קבלת חשיבות הפיצ'רים"""
        
        if not self.is_trained:
            return {}
        
        try:
            importance = self.model.get_feature_importance()
            return dict(zip(self.feature_names, importance))
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Matchmaking Ranker")
    parser.add_argument("--train", action="store_true", help="Train the ranker model")
    args = parser.parse_args()
    
    if args.train:
        print("Training ranker model...")
        ranker = MatchingRanker()
        
        # כאן תצטרך לטעון נתוני אימון אמיתיים
        print("Note: Need to load actual training data for training")
        print("Ranker module ready for training when data is available")
