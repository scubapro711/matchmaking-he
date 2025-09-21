"""
מסנני תנאי סף (Hard Constraints) למערכת השידוכים
"""
from typing import List, Tuple
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from src.data_schemas import Candidate, Preferences, HardConstraints
import logging

logger = logging.getLogger(__name__)

class RulesFilter:
    """מחלקה לסינון מועמדים על פי תנאי סף מחייבים"""
    
    def __init__(self):
        self.geocoder = Nominatim(user_agent="matchmaking-he")
        self._location_cache = {}
    
    def get_coordinates(self, location: str) -> Tuple[float, float]:
        """קבלת קואורדינטות של מיקום"""
        if location in self._location_cache:
            return self._location_cache[location]
        
        try:
            location_data = self.geocoder.geocode(f"{location}, Israel")
            if location_data:
                coords = (location_data.latitude, location_data.longitude)
                self._location_cache[location] = coords
                return coords
        except Exception as e:
            logger.warning(f"Failed to geocode {location}: {e}")
        
        # ברירת מחדל - תל אביב
        default_coords = (32.0853, 34.7818)
        self._location_cache[location] = default_coords
        return default_coords
    
    def calculate_distance(self, location1: str, location2: str) -> float:
        """חישוב מרחק בין שני מיקומים בק"מ"""
        try:
            coords1 = self.get_coordinates(location1)
            coords2 = self.get_coordinates(location2)
            return geodesic(coords1, coords2).kilometers
        except Exception as e:
            logger.warning(f"Failed to calculate distance: {e}")
            return 0.0
    
    def check_age_constraint(self, candidate: Candidate, constraints: HardConstraints) -> bool:
        """בדיקת תנאי גיל"""
        if constraints.min_age and candidate.age < constraints.min_age:
            return False
        if constraints.max_age and candidate.age > constraints.max_age:
            return False
        return True
    
    def check_distance_constraint(self, candidate: Candidate, requester_location: str, 
                                constraints: HardConstraints) -> bool:
        """בדיקת תנאי מרחק גיאוגרפי"""
        if not constraints.max_distance_km:
            return True
        
        distance = self.calculate_distance(candidate.location, requester_location)
        return distance <= constraints.max_distance_km
    
    def check_smoking_constraint(self, candidate: Candidate, constraints: HardConstraints) -> bool:
        """בדיקת תנאי עישון"""
        if constraints.smoking is None:
            return True
        return candidate.smoking == constraints.smoking
    
    def check_community_constraint(self, candidate: Candidate, constraints: HardConstraints) -> bool:
        """בדיקת תנאי קהילה/זרם"""
        if not constraints.required_communities:
            return True
        return candidate.community in constraints.required_communities
    
    def check_religiosity_constraint(self, candidate: Candidate, constraints: HardConstraints) -> bool:
        """בדיקת תנאי רמת דתיות"""
        if not constraints.required_religiosity:
            return True
        return candidate.religiosity_level in constraints.required_religiosity
    
    def check_language_constraint(self, candidate: Candidate, constraints: HardConstraints) -> bool:
        """בדיקת תנאי שפות"""
        if not constraints.required_languages:
            return True
        return any(lang in candidate.languages for lang in constraints.required_languages)
    
    def is_valid_match(self, candidate: Candidate, requester: Candidate, 
                      preferences: Preferences) -> Tuple[bool, str]:
        """
        בדיקה מקיפה האם מועמד עומד בכל תנאי הסף
        
        Returns:
            Tuple[bool, str]: (האם עובר, סיבת פסילה אם לא עובר)
        """
        constraints = preferences.must_have
        
        # בדיקת גיל
        if not self.check_age_constraint(candidate, constraints):
            return False, f"גיל {candidate.age} לא עומד בתנאי הגיל הנדרש"
        
        # בדיקת מרחק
        if not self.check_distance_constraint(candidate, requester.location, constraints):
            distance = self.calculate_distance(candidate.location, requester.location)
            return False, f"מרחק {distance:.1f} ק\"מ חורג מהמרחק המותר"
        
        # בדיקת עישון
        if not self.check_smoking_constraint(candidate, constraints):
            return False, "תנאי עישון לא מתאים"
        
        # בדיקת קהילה
        if not self.check_community_constraint(candidate, constraints):
            return False, f"קהילה {candidate.community.value} לא מתאימה"
        
        # בדיקת רמת דתיות
        if not self.check_religiosity_constraint(candidate, constraints):
            return False, f"רמת דתיות {candidate.religiosity_level.value} לא מתאימה"
        
        # בדיקת שפות
        if not self.check_language_constraint(candidate, constraints):
            return False, "שפות לא מתאימות"
        
        return True, "עובר את כל תנאי הסף"
    
    def filter_candidates(self, candidates: List[Candidate], requester: Candidate, 
                         preferences: Preferences) -> List[Tuple[Candidate, str]]:
        """
        סינון רשימת מועמדים על פי תנאי סף
        
        Returns:
            List[Tuple[Candidate, str]]: רשימת מועמדים שעברו + הסבר
        """
        valid_candidates = []
        
        for candidate in candidates:
            # לא להציע את המועמד לעצמו
            if candidate.id == requester.id:
                continue
            
            # בדיקת מגדר מנוגד (בהנחה שמחפשים מגדר מנוגד)
            if candidate.gender == requester.gender:
                continue
            
            is_valid, reason = self.is_valid_match(candidate, requester, preferences)
            if is_valid:
                valid_candidates.append((candidate, reason))
        
        logger.info(f"Filtered {len(candidates)} candidates to {len(valid_candidates)} valid matches")
        return valid_candidates
