import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering:
    """
    Content-based filtering recommendation system using item features.
    """
    
    def __init__(self, item_features=None):
        """
        Initialize the content-based filtering model.
        
        Args:
            item_features (DataFrame): DataFrame containing item features
        """
        self.item_features = item_features
        self.item_similarity = None
        
    def fit(self, item_features=None):
        """
        Fit the content-based filtering model to the item features.
        
        Args:
            item_features (DataFrame, optional): DataFrame containing item features
            
        Returns:
            self: The fitted model
        """
        if item_features is not None:
            self.item_features = item_features
            
        if self.item_features is None:
            raise ValueError("Item features not provided")
            
        # Calculate item similarity matrix based on features
        self.item_similarity = cosine_similarity(self.item_features)
        
        return self
    
    def recommend_similar_items(self, item_id, n_recommendations=5):
        """
        Recommend items similar to a given item.
        
        Args:
            item_id: ID of the item
            n_recommendations (int): Number of recommendations to generate
            
        Returns:
            DataFrame: Similar items with similarity scores
        """
        if self.item_similarity is None:
            self.fit()
            
        # Get the item's index in the feature matrix
        if item_id in self.item_features.index:
            item_idx = self.item_features.index.get_loc(item_id)
        else:
            raise ValueError(f"Item ID {item_id} not found in the item features")
            
        # Get similarity scores for the item
        item_sim_scores = self.item_similarity[item_idx]
        
        # Create a DataFrame with similarity scores
        similar_items = pd.DataFrame({
            'item_id': self.item_features.index,
            'similarity': item_sim_scores
        })
        
        # Remove the item itself
        similar_items = similar_items[similar_items['item_id'] != item_id]
        
        # Sort by similarity and take top n
        similar_items = similar_items.sort_values('similarity', ascending=False).head(n_recommendations)
        
        return similar_items
    
    def recommend_for_user(self, user_ratings, n_recommendations=5):
        """
        Recommend items for a user based on their rating history and item content.
        
        Args:
            user_ratings (DataFrame): DataFrame containing user's ratings with item_id and rating columns
            n_recommendations (int): Number of recommendations to generate
            
        Returns:
            DataFrame: Recommended items with predicted scores
        """
        if self.item_similarity is None:
            self.fit()
            
        # Get the items the user has rated
        rated_items = user_ratings['item_id'].values
        
        # Get the indices of rated items in the feature matrix
        rated_indices = [self.item_features.index.get_loc(item_id) 
                         for item_id in rated_items 
                         if item_id in self.item_features.index]
        
        if not rated_indices:
            raise ValueError("None of the user's rated items found in the item features")
            
        # Get the user's ratings for these items
        ratings = user_ratings.set_index('item_id').loc[
            [self.item_features.index[idx] for idx in rated_indices]
        ]['rating'].values
        
        # Normalize ratings to range [0, 1]
        min_rating = min(ratings)
        max_rating = max(ratings)
        if max_rating > min_rating:
            normalized_ratings = (ratings - min_rating) / (max_rating - min_rating)
        else:
            normalized_ratings = np.ones_like(ratings)
        
        # Calculate weighted similarity scores for all items
        weighted_scores = np.zeros(len(self.item_features))
        
        for i, rating in zip(rated_indices, normalized_ratings):
            weighted_scores += self.item_similarity[i] * rating
            
        # Create a DataFrame with weighted scores
        recommendations = pd.DataFrame({
            'item_id': self.item_features.index,
            'predicted_score': weighted_scores
        })
        
        # Remove items the user has already rated
        recommendations = recommendations[~recommendations['item_id'].isin(rated_items)]
        
        # Sort by predicted score and take top n
        recommendations = recommendations.sort_values('predicted_score', ascending=False).head(n_recommendations)
        
        return recommendations
    
    def get_item_profile(self, item_id):
        """
        Get the feature profile of a specific item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            Series: Item feature vector
        """
        if item_id in self.item_features.index:
            return self.item_features.loc[item_id]
        else:
            raise ValueError(f"Item ID {item_id} not found in the item features")