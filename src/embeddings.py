"""
מודול אמבדינגים סמנטיים בעברית למערכת השידוכים
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class HebrewEmbeddings:
    """מחלקה לטיפול באמבדינגים סמנטיים בעברית"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        אתחול מודל האמבדינגים
        
        Args:
            model_name: שם המודל הרב-לשוני שתומך בעברית
        """
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        self.cache_file = "models/embeddings_cache.pkl"
        
        # יצירת תיקיית מודלים אם לא קיימת
        os.makedirs("models", exist_ok=True)
        
        self._load_model()
        self._load_cache()
    
    def _load_model(self):
        """טעינת מודל האמבדינגים"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_cache(self):
        """טעינת מטמון האמבדינגים"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embeddings_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {e}")
                self.embeddings_cache = {}
    
    def _save_cache(self):
        """שמירת מטמון האמבדינגים"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            logger.info(f"Saved {len(self.embeddings_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embeddings cache: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """עיבוד מקדים של טקסט עברי"""
        if not text:
            return ""
        
        # ניקוי בסיסי
        text = text.strip()
        
        # הסרת תווים מיוחדים מיותרים
        text = text.replace('\n', ' ').replace('\t', ' ')
        
        # הסרת רווחים כפולים
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        return text
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        קבלת אמבדינג לטקסט
        
        Args:
            text: הטקסט לעיבוד
            
        Returns:
            np.ndarray: וקטור האמבדינג
        """
        if not text:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        processed_text = self.preprocess_text(text)
        
        # בדיקה במטמון
        if processed_text in self.embeddings_cache:
            return self.embeddings_cache[processed_text]
        
        try:
            # יצירת אמבדינג חדש
            embedding = self.model.encode(processed_text, convert_to_numpy=True)
            
            # שמירה במטמון
            self.embeddings_cache[processed_text] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to create embedding for text: {e}")
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        קבלת אמבדינגים לרשימת טקסטים
        
        Args:
            texts: רשימת טקסטים
            
        Returns:
            List[np.ndarray]: רשימת וקטורי אמבדינג
        """
        embeddings = []
        new_texts = []
        new_indices = []
        
        # בדיקה מה כבר קיים במטמון
        for i, text in enumerate(texts):
            processed_text = self.preprocess_text(text)
            if processed_text in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[processed_text])
            else:
                embeddings.append(None)
                new_texts.append(processed_text)
                new_indices.append(i)
        
        # יצירת אמבדינגים חדשים
        if new_texts:
            try:
                new_embeddings = self.model.encode(new_texts, convert_to_numpy=True)
                
                # עדכון המטמון והתוצאות
                for idx, embedding in zip(new_indices, new_embeddings):
                    processed_text = new_texts[new_indices.index(idx)]
                    self.embeddings_cache[processed_text] = embedding
                    embeddings[idx] = embedding
                    
            except Exception as e:
                logger.error(f"Failed to create batch embeddings: {e}")
                # מילוי באפסים במקרה של שגיאה
                for idx in new_indices:
                    embeddings[idx] = np.zeros(self.model.get_sentence_embedding_dimension())
        
        return embeddings
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        חישוב דמיון קוסינוס בין שני טקסטים
        
        Args:
            text1, text2: הטקסטים להשוואה
            
        Returns:
            float: ציון דמיון (0-1)
        """
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        # חישוב דמיון קוסינוס
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        
        # וידוא שהתוצאה בטווח 0-1
        return max(0.0, min(1.0, similarity))
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str], 
                         top_k: int = 10) -> List[Tuple[int, float]]:
        """
        מציאת הטקסטים הדומים ביותר לשאילתה
        
        Args:
            query_text: טקסט השאילתה
            candidate_texts: רשימת טקסטים למציאת דמיון
            top_k: מספר התוצאות המובילות
            
        Returns:
            List[Tuple[int, float]]: רשימת (אינדקס, ציון דמיון)
        """
        query_embedding = self.get_embedding(query_text)
        candidate_embeddings = self.get_embeddings_batch(candidate_texts)
        
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = cosine_similarity([query_embedding], [candidate_embedding])[0][0]
            similarities.append((i, max(0.0, min(1.0, similarity))))
        
        # מיון לפי ציון דמיון
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def build_embeddings_for_candidates(self, candidates_data: List[Dict]) -> Dict[str, np.ndarray]:
        """
        בניית אמבדינגים לכל המועמדים
        
        Args:
            candidates_data: רשימת נתוני מועמדים
            
        Returns:
            Dict[str, np.ndarray]: מילון של ID מועמד -> אמבדינג
        """
        logger.info(f"Building embeddings for {len(candidates_data)} candidates")
        
        candidate_embeddings = {}
        texts_to_process = []
        candidate_ids = []
        
        for candidate in candidates_data:
            # שילוב תיאור אישי והעדפות
            combined_text = f"{candidate.get('description_text', '')} {candidate.get('preferences_text', '')}"
            texts_to_process.append(combined_text)
            candidate_ids.append(candidate['id'])
        
        # יצירת אמבדינגים
        embeddings = self.get_embeddings_batch(texts_to_process)
        
        # יצירת מילון
        for candidate_id, embedding in zip(candidate_ids, embeddings):
            candidate_embeddings[candidate_id] = embedding
        
        # שמירת מטמון
        self._save_cache()
        
        logger.info(f"Successfully built embeddings for {len(candidate_embeddings)} candidates")
        return candidate_embeddings

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hebrew Embeddings Builder")
    parser.add_argument("--build", action="store_true", help="Build embeddings for sample data")
    args = parser.parse_args()
    
    if args.build:
        # דוגמה לבניית אמבדינגים
        embeddings = HebrewEmbeddings()
        
        sample_texts = [
            "אני מחפש בן זוג דתי ומשפחתי שאוהב ללמוד תורה",
            "מעוניינת בבחור עם השכלה גבוהה ויחס חם למשפחה",
            "חשוב לי מישהו שמבין את החשיבות של מסורת ומודרניות"
        ]
        
        print("Building sample embeddings...")
        for i, text in enumerate(sample_texts):
            embedding = embeddings.get_embedding(text)
            print(f"Text {i+1}: {len(embedding)} dimensions")
        
        print("Embeddings built successfully!")
