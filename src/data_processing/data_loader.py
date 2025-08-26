import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Class for loading and preprocessing data for the recommendation system.
    """
    
    def __init__(self, ratings_path, products_path=None):
        """
        Initialize the DataLoader with paths to data files.
        
        Args:
            ratings_path (str): Path to the ratings data file
            products_path (str, optional): Path to the products data file
        """
        self.ratings_path = ratings_path
        self.products_path = products_path
        self.ratings_data = None
        self.products_data = None
        self.user_item_matrix = None
        
    def load_data(self):
        """
        Load ratings and products data from CSV files.
        
        Returns:
            tuple: (ratings_df, products_df) - DataFrames containing the loaded data
        """
        self.ratings_data = pd.read_csv(self.ratings_path)
        
        if self.products_path and os.path.exists(self.products_path):
            self.products_data = pd.read_csv(self.products_path)
        
        return self.ratings_data, self.products_data
    
    def preprocess_ratings(self):
        """
        Preprocess ratings data by handling missing values and duplicates.
        
        Returns:
            DataFrame: Preprocessed ratings data
        """
        if self.ratings_data is None:
            self.load_data()
            
        # Handle missing values
        self.ratings_data = self.ratings_data.dropna(subset=['user_id', 'item_id', 'rating'])
        
        # Handle duplicates (keep the most recent rating)
        if 'timestamp' in self.ratings_data.columns:
            self.ratings_data = self.ratings_data.sort_values('timestamp').drop_duplicates(
                subset=['user_id', 'item_id'], keep='last'
            )
        
        return self.ratings_data
    
    def create_user_item_matrix(self):
        """
        Create a user-item matrix from the ratings data.
        
        Returns:
            DataFrame: User-item matrix where rows are users, columns are items,
                      and values are ratings
        """
        if self.ratings_data is None:
            self.preprocess_ratings()
            
        # Create the user-item matrix
        self.user_item_matrix = self.ratings_data.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        return self.user_item_matrix
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the ratings data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_df, test_df) - Training and testing DataFrames
        """
        if self.ratings_data is None:
            self.preprocess_ratings()
            
        train_df, test_df = train_test_split(
            self.ratings_data,
            test_size=test_size,
            random_state=random_state,
            stratify=self.ratings_data['user_id'] if len(self.ratings_data['user_id'].unique()) < 10 else None
        )
        
        return train_df, test_df
    
    def get_user_profile(self, user_id):
        """
        Get the profile of a specific user based on their ratings.
        
        Args:
            user_id: ID of the user
            
        Returns:
            DataFrame: User's ratings
        """
        if self.ratings_data is None:
            self.preprocess_ratings()
            
        return self.ratings_data[self.ratings_data['user_id'] == user_id]
    
    def get_item_profile(self, item_id):
        """
        Get the profile of a specific item based on its ratings.
        
        Args:
            item_id: ID of the item
            
        Returns:
            tuple: (ratings, item_data) - Item's ratings and metadata
        """
        if self.ratings_data is None:
            self.preprocess_ratings()
            
        item_ratings = self.ratings_data[self.ratings_data['item_id'] == item_id]
        
        item_data = None
        if self.products_data is not None:
            item_data = self.products_data[self.products_data['item_id'] == item_id]
            
        return item_ratings, item_data