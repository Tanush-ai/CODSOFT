import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    """
    Class for extracting features from user and item data for recommendation.
    """
    
    def __init__(self, ratings_df=None, products_df=None):
        """
        Initialize the FeatureExtractor with ratings and products data.
        
        Args:
            ratings_df (DataFrame): DataFrame containing user ratings
            products_df (DataFrame): DataFrame containing product information
        """
        self.ratings_df = ratings_df
        self.products_df = products_df
        self.user_features = None
        self.item_features = None
        
    def extract_user_features(self):
        """
        Extract features for users based on their rating behavior.
        
        Returns:
            DataFrame: User features
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data not provided")
            
        # Calculate basic user statistics
        user_stats = self.ratings_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max']
        })
        
        user_stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'min_rating', 'max_rating']
        user_stats = user_stats.fillna(0)  # Fill NaN values (e.g., if std is NaN due to single rating)
        
        # Calculate category preferences if product data is available
        if self.products_df is not None:
            # Merge ratings with product data to get category information
            merged_data = pd.merge(
                self.ratings_df, 
                self.products_df[['item_id', 'category']], 
                on='item_id'
            )
            
            # Calculate average rating per category for each user
            category_prefs = merged_data.groupby(['user_id', 'category'])['rating'].mean().unstack().fillna(0)
            category_prefs.columns = [f'pref_{col}' for col in category_prefs.columns]
            
            # Join category preferences with user stats
            user_features = user_stats.join(category_prefs, how='left').fillna(0)
        else:
            user_features = user_stats
            
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(user_features)
        self.user_features = pd.DataFrame(
            scaled_features, 
            index=user_features.index, 
            columns=user_features.columns
        )
        
        return self.user_features
    
    def extract_item_features(self):
        """
        Extract features for items based on their attributes and rating patterns.
        
        Returns:
            DataFrame: Item features
        """
        if self.products_df is None:
            raise ValueError("Product data not provided")
            
        # Calculate basic item statistics from ratings
        if self.ratings_df is not None:
            item_stats = self.ratings_df.groupby('item_id').agg({
                'rating': ['count', 'mean', 'std']
            })
            
            item_stats.columns = ['rating_count', 'avg_rating', 'rating_std']
            item_stats = item_stats.fillna(0)
        else:
            item_stats = pd.DataFrame(index=self.products_df['item_id'])
            
        # One-hot encode categorical features
        if 'category' in self.products_df.columns:
            category_dummies = pd.get_dummies(self.products_df['category'], prefix='category')
            category_features = pd.concat([
                self.products_df[['item_id']], 
                category_dummies
            ], axis=1).set_index('item_id')
        else:
            category_features = pd.DataFrame(index=self.products_df['item_id'])
            
        # Extract text features if description is available
        if 'description' in self.products_df.columns:
            # Use TF-IDF to extract features from product descriptions
            tfidf = TfidfVectorizer(max_features=10, stop_words='english')
            description_features = tfidf.fit_transform(
                self.products_df['description'].fillna('')
            )
            
            # Convert to DataFrame
            description_df = pd.DataFrame(
                description_features.toarray(),
                index=self.products_df['item_id'],
                columns=[f'desc_{i}' for i in range(description_features.shape[1])]
            )
        else:
            description_df = pd.DataFrame(index=self.products_df['item_id'])
            
        # Combine all features
        numerical_features = self.products_df[['item_id', 'price']] if 'price' in self.products_df.columns else self.products_df[['item_id']]
        numerical_features = numerical_features.set_index('item_id')
        
        # Join all feature sets
        item_features = item_stats.join(
            [category_features, description_df, numerical_features], 
            how='outer'
        ).fillna(0)
        
        # Normalize numerical features
        numerical_cols = ['price', 'avg_rating', 'rating_count', 'rating_std']
        numerical_cols = [col for col in numerical_cols if col in item_features.columns]
        
        if numerical_cols:
            scaler = StandardScaler()
            item_features[numerical_cols] = scaler.fit_transform(item_features[numerical_cols])
            
        self.item_features = item_features
        return self.item_features
    
    def get_user_profile_vector(self, user_id):
        """
        Get the feature vector for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Series: User feature vector
        """
        if self.user_features is None:
            self.extract_user_features()
            
        if user_id in self.user_features.index:
            return self.user_features.loc[user_id]
        else:
            # Return default features for new users
            return pd.Series(0, index=self.user_features.columns)
    
    def get_item_profile_vector(self, item_id):
        """
        Get the feature vector for a specific item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            Series: Item feature vector
        """
        if self.item_features is None:
            self.extract_item_features()
            
        if item_id in self.item_features.index:
            return self.item_features.loc[item_id]
        else:
            # Return default features for new items
            return pd.Series(0, index=self.item_features.columns)