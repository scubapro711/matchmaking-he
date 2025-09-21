"""
מודול איחוד מאגרי נתונים ובדיקת כפילויות
מיועד לאיחוד המאגר הקיים עם המאגר החדש עם נתוני מוצא
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class DataMerger:
    """מחלקה לאיחוד מאגרי נתונים ובדיקת כפילויות"""
    
    def __init__(self):
        # מיפוי עמודות בין המאגרים השונים
        self.column_mapping = {
            # מאגר ישן -> מאגר חדש
            'שם פרטי': 'שם_פרטי',
            'שם משפחה': 'שם_משפחה',
            'טלפון נייד': 'טלפון_נייד',
            'מגזר דתי': 'מגזר_דתי',
            'ישיבה/סמינר': 'ישיבה_סמינר',
            'סטטוס משפחתי': 'סטטוס_משפחתי',
            'מה מחפש/ת': 'העדפות',
            'הערות': 'הערות'
        }
        
        # מיפוי מגזרים דתיים לנרמול
        self.religious_normalization = {
            'ליטאי': 'חרדי_ליטאי',
            'חסידי': 'חרדי_חסידי',
            'ספרדי': 'חרדי_ספרדי',
            'דת"ל': 'דתי_לאומי',
            'דתי לאומי': 'דתי_לאומי',
            'בע"ת': 'בעל_תשובה',
            'מסורתי': 'מסורתי'
        }
        
        # מיפוי עדות לנרמול
        self.origin_normalization = {
            'אשכנזי': 'אשכנזי',
            'ספרדי': 'ספרדי',
            'מרוקו': 'מרוקו',
            'תימן': 'תימן',
            'עיראק': 'עיראק',
            'פרס': 'פרס',
            'תוניס': 'תוניס',
            'חלב': 'חלב',
            'מזרחי': 'מזרחי',
            'מעורב': 'מעורב'
        }
    
    def load_original_data(self, file_path: str) -> pd.DataFrame:
        """טעינת המאגר המקורי"""
        logger.info(f"Loading original data from {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records from original data")
        
        # נרמול עמודות
        df_normalized = self._normalize_original_columns(df)
        
        return df_normalized
    
    def load_new_data(self, file_path: str) -> pd.DataFrame:
        """טעינת המאגר החדש עם נתוני מוצא"""
        logger.info(f"Loading new data with origins from {file_path}")
        
        # מציאת שורת הכותרות
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        header_line = None
        for i, line in enumerate(lines):
            if 'מזהה' in line and 'שם_פרטי' in line:
                header_line = i
                break
        
        if header_line is None:
            raise ValueError("Could not find header line in new data file")
        
        # קריאת הנתונים
        df = pd.read_csv(file_path, skiprows=header_line, encoding='utf-8')
        
        # הסרת שורות ריקות
        df = df.dropna(how='all')
        
        # הסרת עמודות לא רלוונטיות
        relevant_columns = [
            'מזהה', 'שם_פרטי', 'שם_משפחה', 'טלפון_נייד', 'מקור_קובץ',
            'מין', 'גיל', 'עיר', 'מגזר_דתי', 'גובה', 'עיסוק', 'ישיבה_סמינר',
            'הערות', 'עדת_אב', 'עדת_אם', 'הסבר_מוצא', 'רמת_ביטחון_מוצא',
            'תובנות_AI'
        ]
        
        # שמירת עמודות קיימות בלבד
        existing_columns = [col for col in relevant_columns if col in df.columns]
        df = df[existing_columns]
        
        logger.info(f"Loaded {len(df)} records from new data")
        
        return df
    
    def _normalize_original_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """נרמול עמודות המאגר המקורי"""
        
        # שינוי שמות עמודות
        df_renamed = df.rename(columns=self.column_mapping)
        
        # הוספת עמודות חסרות
        if 'עדת_אב' not in df_renamed.columns:
            df_renamed['עדת_אב'] = self._extract_origin_from_religious(df_renamed.get('מגזר_דתי', ''))
        
        if 'עדת_אם' not in df_renamed.columns:
            df_renamed['עדת_אם'] = df_renamed['עדת_אב']  # ברירת מחדל
        
        if 'רמת_ביטחון_מוצא' not in df_renamed.columns:
            df_renamed['רמת_ביטחון_מוצא'] = 'ניחוש מהמגזר הדתי'
        
        if 'הסבר_מוצא' not in df_renamed.columns:
            df_renamed['הסבר_מוצא'] = 'נקבע לפי המגזר הדתי במאגר המקורי'
        
        if 'מקור_קובץ' not in df_renamed.columns:
            df_renamed['מקור_קובץ'] = 'מאגר_מקורי'
        
        return df_renamed
    
    def _extract_origin_from_religious(self, religious_series) -> pd.Series:
        """חילוץ מוצא מהמגזר הדתי"""
        
        def extract_origin(religious_value):
            if pd.isna(religious_value):
                return 'לא_מוגדר'
            
            religious_str = str(religious_value).lower()
            
            if 'ספרדי' in religious_str:
                return 'ספרדי'
            elif 'אשכנזי' in religious_str or 'ליטאי' in religious_str or 'חסידי' in religious_str:
                return 'אשכנזי'
            elif 'מזרחי' in religious_str:
                return 'מזרחי'
            else:
                return 'לא_מוגדר'
        
        if isinstance(religious_series, pd.Series):
            return religious_series.apply(extract_origin)
        else:
            return extract_origin(religious_series)
    
    def detect_duplicates_between_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """זיהוי כפילויות בין שני מאגרים"""
        logger.info("Detecting duplicates between datasets")
        
        duplicates_info = {
            'by_name': [],
            'by_phone': [],
            'by_name_age': [],
            'potential_matches': []
        }
        
        # כפילויות לפי שם
        if all(col in df1.columns and col in df2.columns for col in ['שם_פרטי', 'שם_משפחה']):
            for _, row1 in df1.iterrows():
                name1 = f"{row1['שם_פרטי']} {row1['שם_משפחה']}".strip().lower()
                
                for _, row2 in df2.iterrows():
                    name2 = f"{row2['שם_פרטי']} {row2['שם_משפחה']}".strip().lower()
                    
                    if name1 == name2 and name1 != '':
                        duplicates_info['by_name'].append({
                            'df1_id': row1.get('מזהה', 'unknown'),
                            'df2_id': row2.get('מזהה', 'unknown'),
                            'name': name1,
                            'df1_age': row1.get('גיל', 'unknown'),
                            'df2_age': row2.get('גיל', 'unknown')
                        })
        
        # כפילויות לפי טלפון
        if 'טלפון_נייד' in df1.columns and 'טלפון_נייד' in df2.columns:
            phones1 = set(df1['טלפון_נייד'].dropna().astype(str))
            phones2 = set(df2['טלפון_נייד'].dropna().astype(str))
            
            common_phones = phones1.intersection(phones2)
            for phone in common_phones:
                if phone and phone != '' and phone != 'nan':
                    duplicates_info['by_phone'].append(phone)
        
        # כפילויות לפי שם + גיל
        if all(col in df1.columns and col in df2.columns for col in ['שם_פרטי', 'שם_משפחה', 'גיל']):
            for _, row1 in df1.iterrows():
                name_age1 = f"{row1['שם_פרטי']} {row1['שם_משפחה']} {row1['גיל']}".strip().lower()
                
                for _, row2 in df2.iterrows():
                    name_age2 = f"{row2['שם_פרטי']} {row2['שם_משפחה']} {row2['גיל']}".strip().lower()
                    
                    if name_age1 == name_age2 and name_age1 != '  ':
                        duplicates_info['by_name_age'].append({
                            'df1_id': row1.get('מזהה', 'unknown'),
                            'df2_id': row2.get('מזהה', 'unknown'),
                            'name_age': name_age1
                        })
        
        logger.info(f"Found duplicates: {len(duplicates_info['by_name'])} by name, "
                   f"{len(duplicates_info['by_phone'])} by phone, "
                   f"{len(duplicates_info['by_name_age'])} by name+age")
        
        return duplicates_info
    
    def remove_duplicates_within_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """הסרת כפילויות בתוך מאגר יחיד"""
        logger.info(f"Removing duplicates within dataset of {len(df)} records")
        
        initial_count = len(df)
        
        # הסרת כפילויות מדויקות לפי מזהה
        if 'מזהה' in df.columns:
            df = df.drop_duplicates(subset=['מזהה'], keep='first')
        
        # הסרת כפילויות לפי שם + גיל + טלפון
        if all(col in df.columns for col in ['שם_פרטי', 'שם_משפחה', 'גיל', 'טלפון_נייד']):
            df = df.drop_duplicates(subset=['שם_פרטי', 'שם_משפחה', 'גיל', 'טלפון_נייד'], keep='first')
        
        # הסרת כפילויות לפי שם + גיל (למקרים שהטלפון שונה)
        if all(col in df.columns for col in ['שם_פרטי', 'שם_משפחה', 'גיל']):
            df = df.drop_duplicates(subset=['שם_פרטי', 'שם_משפחה', 'גיל'], keep='first')
        
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} duplicates ({removed_count/initial_count*100:.1f}%)")
        
        return df
    
    def merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                      remove_cross_duplicates: bool = True) -> pd.DataFrame:
        """איחוד שני מאגרים"""
        logger.info("Merging datasets")
        
        # הסרת כפילויות בתוך כל מאגר
        df1_clean = self.remove_duplicates_within_dataset(df1)
        df2_clean = self.remove_duplicates_within_dataset(df2)
        
        # זיהוי כפילויות בין המאגרים
        if remove_cross_duplicates:
            duplicates_info = self.detect_duplicates_between_datasets(df1_clean, df2_clean)
            
            # הסרת כפילויות מהמאגר השני (שמירת המאגר הראשון)
            if duplicates_info['by_name']:
                duplicate_ids = [dup['df2_id'] for dup in duplicates_info['by_name']]
                df2_clean = df2_clean[~df2_clean['מזהה'].isin(duplicate_ids)]
                logger.info(f"Removed {len(duplicate_ids)} cross-dataset duplicates")
        
        # איחוד המאגרים
        # וידוא שיש עמודות משותפות
        common_columns = list(set(df1_clean.columns).intersection(set(df2_clean.columns)))
        
        # הוספת עמודות חסרות עם ערכי ברירת מחדל
        for col in df1_clean.columns:
            if col not in df2_clean.columns:
                df2_clean[col] = None
        
        for col in df2_clean.columns:
            if col not in df1_clean.columns:
                df1_clean[col] = None
        
        # איחוד
        merged_df = pd.concat([df1_clean, df2_clean], ignore_index=True, sort=False)
        
        # הוספת מזהה ייחודי למקרה שחסר
        if 'מזהה' not in merged_df.columns or merged_df['מזהה'].isna().any():
            merged_df['מזהה'] = merged_df['מזהה'].fillna('').astype(str)
            missing_ids = merged_df['מזהה'] == ''
            merged_df.loc[missing_ids, 'מזהה'] = [f"merged_{i}" for i in range(missing_ids.sum())]
        
        # הוספת תאריך איחוד
        merged_df['תאריך_איחוד'] = datetime.now().isoformat()
        
        logger.info(f"Merged dataset contains {len(merged_df)} records")
        
        return merged_df
    
    def generate_merge_report(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                            merged_df: pd.DataFrame, duplicates_info: Dict) -> Dict:
        """יצירת דוח איחוד מפורט"""
        
        report = {
            'original_datasets': {
                'dataset_1': {
                    'count': len(df1),
                    'columns': list(df1.columns)
                },
                'dataset_2': {
                    'count': len(df2),
                    'columns': list(df2.columns)
                }
            },
            'merged_dataset': {
                'count': len(merged_df),
                'columns': list(merged_df.columns),
                'unique_candidates': len(merged_df)
            },
            'duplicates_found': {
                'by_name': len(duplicates_info.get('by_name', [])),
                'by_phone': len(duplicates_info.get('by_phone', [])),
                'by_name_age': len(duplicates_info.get('by_name_age', []))
            },
            'data_quality': {
                'missing_names': merged_df[['שם_פרטי', 'שם_משפחה']].isna().any(axis=1).sum(),
                'missing_ages': merged_df['גיל'].isna().sum() if 'גיל' in merged_df.columns else 0,
                'missing_phones': merged_df['טלפון_נייד'].isna().sum() if 'טלפון_נייד' in merged_df.columns else 0,
                'missing_origins': merged_df['עדת_אב'].isna().sum() if 'עדת_אב' in merged_df.columns else 0
            },
            'distributions': {
                'gender': merged_df['מין'].value_counts().to_dict() if 'מין' in merged_df.columns else {},
                'religious_sector': merged_df['מגזר_דתי'].value_counts().head(10).to_dict() if 'מגזר_דתי' in merged_df.columns else {},
                'father_origin': merged_df['עדת_אב'].value_counts().head(10).to_dict() if 'עדת_אב' in merged_df.columns else {}
            },
            'merge_timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def save_merged_data(self, merged_df: pd.DataFrame, output_path: str):
        """שמירת הנתונים המאוחדים"""
        logger.info(f"Saving merged data to {output_path}")
        
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved {len(merged_df)} records to {output_path}")

if __name__ == "__main__":
    # דוגמה לשימוש
    merger = DataMerger()
    
    # טעינת המאגרים
    df1 = merger.load_original_data('/path/to/original_data.csv')
    df2 = merger.load_new_data('/path/to/new_data_with_origins.csv')
    
    # זיהוי כפילויות
    duplicates_info = merger.detect_duplicates_between_datasets(df1, df2)
    
    # איחוד
    merged_df = merger.merge_datasets(df1, df2)
    
    # יצירת דוח
    report = merger.generate_merge_report(df1, df2, merged_df, duplicates_info)
    
    # שמירה
    merger.save_merged_data(merged_df, '/path/to/merged_data.csv')
    
    print("=== דוח איחוד ===")
    print(f"מאגר מקורי: {report['original_datasets']['dataset_1']['count']} מועמדים")
    print(f"מאגר חדש: {report['original_datasets']['dataset_2']['count']} מועמדים")
    print(f"מאגר מאוחד: {report['merged_dataset']['count']} מועמדים")
    print(f"כפילויות שהוסרו: {report['duplicates_found']}")
