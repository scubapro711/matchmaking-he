    def load_and_clean_data_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """ניקוי נתונים מ-DataFrame קיים"""
        logger.info(f"Processing DataFrame with {len(df)} records")
        
        # ניקוי בסיסי
        df = self.remove_duplicates(df)
        df = self.normalize_religious_sectors(df)
        df = self.normalize_cities(df)
        df = self.clean_phone_numbers(df)
        df = self.extract_preferences_from_notes(df)
        df = self.validate_ages(df)
        df = self.normalize_heights(df)
        
        logger.info(f"After cleaning: {len(df)} records")
        return df

    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
