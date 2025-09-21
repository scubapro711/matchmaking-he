"""
מודול Stable Matching (Gale-Shapley) למערכת השידוכים
"""
from typing import List, Dict, Tuple, Set, Optional
from src.data_schemas import Candidate, MatchScore
import logging

logger = logging.getLogger(__name__)

class StableMatching:
    """מחלקה ליצירת התאמות יציבות באמצעות אלגוריתם Gale-Shapley"""
    
    def __init__(self):
        self.matches = {}
        self.preferences = {}
        self.reverse_preferences = {}
    
    def build_preference_lists(self, scores: List[MatchScore], 
                             candidates: Dict[str, Candidate]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        בניית רשימות העדפות מציוני התאמה
        
        Args:
            scores: רשימת ציוני התאמה
            candidates: מילון מועמדים
            
        Returns:
            Tuple של (העדפות גברים, העדפות נשים)
        """
        
        # הפרדה לגברים ונשים
        men = {cid: candidate for cid, candidate in candidates.items() 
               if candidate.gender.value == 'M'}
        women = {cid: candidate for cid, candidate in candidates.items() 
                if candidate.gender.value == 'F'}
        
        # אתחול רשימות העדפות
        men_preferences = {man_id: [] for man_id in men.keys()}
        women_preferences = {woman_id: [] for woman_id in women.keys()}
        
        # מיון ציונים לפי מועמד מבקש
        scores_by_requester = {}
        for score in scores:
            requester_id = score.candidate_a_id
            if requester_id not in scores_by_requester:
                scores_by_requester[requester_id] = []
            scores_by_requester[requester_id].append(score)
        
        # בניית רשימות העדפות לגברים
        for man_id in men.keys():
            if man_id in scores_by_requester:
                # מיון לפי ציון (גבוה לנמוך)
                sorted_scores = sorted(scores_by_requester[man_id], 
                                     key=lambda x: x.total_score, reverse=True)
                
                # הוספת נשים בלבד לרשימת ההעדפות
                for score in sorted_scores:
                    target_id = score.candidate_b_id
                    if target_id in women and target_id not in men_preferences[man_id]:
                        men_preferences[man_id].append(target_id)
        
        # בניית רשימות העדפות לנשים
        for woman_id in women.keys():
            if woman_id in scores_by_requester:
                # מיון לפי ציון (גבוה לנמוך)
                sorted_scores = sorted(scores_by_requester[woman_id], 
                                     key=lambda x: x.total_score, reverse=True)
                
                # הוספת גברים בלבד לרשימת ההעדפות
                for score in sorted_scores:
                    target_id = score.candidate_b_id
                    if target_id in men and target_id not in women_preferences[woman_id]:
                        women_preferences[woman_id].append(target_id)
        
        logger.info(f"Built preference lists: {len(men_preferences)} men, {len(women_preferences)} women")
        return men_preferences, women_preferences
    
    def gale_shapley(self, men_preferences: Dict[str, List[str]], 
                    women_preferences: Dict[str, List[str]]) -> Dict[str, str]:
        """
        אלגוריתם Gale-Shapley לחיפוש התאמות יציבות
        
        Args:
            men_preferences: העדפות הגברים
            women_preferences: העדפות הנשים
            
        Returns:
            Dict[str, str]: מיפוי של גבר -> אישה
        """
        
        # אתחול
        men_partner = {}  # גבר -> אישה
        women_partner = {}  # אישה -> גבר
        men_next_proposal = {man: 0 for man in men_preferences.keys()}  # אינדקס ההצעה הבאה
        
        # רשימת גברים חופשיים
        free_men = list(men_preferences.keys())
        
        # יצירת מילון דירוג לנשים (לביצועים טובים יותר)
        women_ranking = {}
        for woman, preferences in women_preferences.items():
            women_ranking[woman] = {man: i for i, man in enumerate(preferences)}
        
        iteration = 0
        max_iterations = len(men_preferences) * len(women_preferences)
        
        while free_men and iteration < max_iterations:
            iteration += 1
            man = free_men.pop(0)
            
            # בדיקה שיש עוד נשים להציע להן
            if men_next_proposal[man] >= len(men_preferences[man]):
                continue
            
            # הנשים הבאה ברשימת ההעדפות של הגבר
            woman = men_preferences[man][men_next_proposal[man]]
            men_next_proposal[man] += 1
            
            # בדיקה שהאישה מכירה את הגבר
            if woman not in women_ranking or man not in women_ranking[woman]:
                free_men.append(man)  # הגבר נשאר חופשי
                continue
            
            # אם האישה חופשייה
            if woman not in women_partner:
                # התאמה
                men_partner[man] = woman
                women_partner[woman] = man
                logger.debug(f"New match: {man} - {woman}")
                
            else:
                # האישה כבר מותאמת - בדיקה מי עדיף
                current_partner = women_partner[woman]
                
                # השוואת דירוגים (נמוך יותר = עדיף יותר)
                current_rank = women_ranking[woman].get(current_partner, float('inf'))
                new_rank = women_ranking[woman].get(man, float('inf'))
                
                if new_rank < current_rank:
                    # הגבר החדש עדיף
                    men_partner[man] = woman
                    women_partner[woman] = man
                    
                    # השותף הקודם הופך לחופשי
                    del men_partner[current_partner]
                    free_men.append(current_partner)
                    
                    logger.debug(f"Better match: {man} - {woman} (replaced {current_partner})")
                else:
                    # השותף הנוכחי עדיף - הגבר נשאר חופשי
                    free_men.append(man)
        
        if iteration >= max_iterations:
            logger.warning("Gale-Shapley reached maximum iterations")
        
        logger.info(f"Gale-Shapley completed in {iteration} iterations. Found {len(men_partner)} matches.")
        return men_partner
    
    def find_stable_matches(self, scores: List[MatchScore], 
                          candidates: Dict[str, Candidate],
                          min_score: float = 0.3) -> List[Tuple[str, str, float]]:
        """
        מציאת התאמות יציבות
        
        Args:
            scores: רשימת ציוני התאמה
            candidates: מילון מועמדים
            min_score: ציון מינימלי להתאמה
            
        Returns:
            List[Tuple[str, str, float]]: רשימת (גבר, אישה, ציון)
        """
        
        # סינון ציונים נמוכים
        filtered_scores = [score for score in scores if score.total_score >= min_score]
        
        if not filtered_scores:
            logger.warning("No scores above minimum threshold")
            return []
        
        # בניית רשימות העדפות
        men_preferences, women_preferences = self.build_preference_lists(filtered_scores, candidates)
        
        if not men_preferences or not women_preferences:
            logger.warning("No valid preferences found")
            return []
        
        # הרצת אלגוריתם Gale-Shapley
        matches = self.gale_shapley(men_preferences, women_preferences)
        
        # יצירת רשימת התאמות עם ציונים
        stable_matches = []
        score_lookup = {(score.candidate_a_id, score.candidate_b_id): score.total_score 
                       for score in filtered_scores}
        
        for man_id, woman_id in matches.items():
            # חיפוש הציון המתאים
            score = score_lookup.get((man_id, woman_id), 0.0)
            if score == 0.0:
                score = score_lookup.get((woman_id, man_id), 0.0)
            
            stable_matches.append((man_id, woman_id, score))
        
        # מיון לפי ציון
        stable_matches.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Found {len(stable_matches)} stable matches")
        return stable_matches
    
    def validate_stability(self, matches: List[Tuple[str, str, float]], 
                          scores: List[MatchScore],
                          candidates: Dict[str, Candidate]) -> bool:
        """
        בדיקת יציבות ההתאמות
        
        Args:
            matches: רשימת התאמות
            scores: ציוני התאמה
            candidates: מועמדים
            
        Returns:
            bool: האם ההתאמות יציבות
        """
        
        # יצירת מיפוי התאמות
        match_dict = {}
        for man, woman, _ in matches:
            match_dict[man] = woman
            match_dict[woman] = man
        
        # יצירת מילון ציונים
        score_dict = {}
        for score in scores:
            score_dict[(score.candidate_a_id, score.candidate_b_id)] = score.total_score
            score_dict[(score.candidate_b_id, score.candidate_a_id)] = score.total_score
        
        # בדיקת יציבות
        for score in scores:
            person_a = score.candidate_a_id
            person_b = score.candidate_b_id
            
            # אם שניהם לא מותאמים, זה לא מפר יציבות
            if person_a not in match_dict or person_b not in match_dict:
                continue
            
            # אם הם מותאמים זה לזה, זה טוב
            if match_dict.get(person_a) == person_b:
                continue
            
            # בדיקה אם יש להם העדפה הדדית על השותפים הנוכחיים
            current_partner_a = match_dict.get(person_a)
            current_partner_b = match_dict.get(person_b)
            
            if current_partner_a and current_partner_b:
                current_score_a = score_dict.get((person_a, current_partner_a), 0)
                current_score_b = score_dict.get((person_b, current_partner_b), 0)
                potential_score = score.total_score
                
                # אם שניהם מעדיפים זה את זה על השותפים הנוכחיים
                if potential_score > current_score_a and potential_score > current_score_b:
                    logger.warning(f"Instability found: {person_a}-{person_b} prefer each other "
                                 f"over current partners")
                    return False
        
        logger.info("All matches are stable")
        return True
