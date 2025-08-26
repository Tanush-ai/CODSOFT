import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RecommendationMetrics:
    """
    Class for evaluating recommendation system performance.
    """
    
    @staticmethod
    def rmse(y_true, y_pred):
        """
        Calculate Root Mean Squared Error.
        
        Args:
            y_true (array-like): True ratings
            y_pred (array-like): Predicted ratings
            
        Returns:
            float: RMSE value
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true, y_pred):
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true (array-like): True ratings
            y_pred (array-like): Predicted ratings
            
        Returns:
            float: MAE value
        """
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def precision_at_k(recommended_items, relevant_items, k=5):
        """
        Calculate Precision@K.
        
        Args:
            recommended_items (list): List of recommended item IDs
            relevant_items (list): List of relevant (true positive) item IDs
            k (int): Number of recommendations to consider
            
        Returns:
            float: Precision@K value
        """
        if len(recommended_items) == 0:
            return 0.0
            
        # Consider only the top-k recommendations
        recommended_items = recommended_items[:k]
        
        # Count the number of relevant items in the recommendations
        relevant_and_recommended = set(relevant_items).intersection(set(recommended_items))
        
        return len(relevant_and_recommended) / min(k, len(recommended_items))
    
    @staticmethod
    def recall_at_k(recommended_items, relevant_items, k=5):
        """
        Calculate Recall@K.
        
        Args:
            recommended_items (list): List of recommended item IDs
            relevant_items (list): List of relevant (true positive) item IDs
            k (int): Number of recommendations to consider
            
        Returns:
            float: Recall@K value
        """
        if len(relevant_items) == 0:
            return 0.0
            
        # Consider only the top-k recommendations
        recommended_items = recommended_items[:k]
        
        # Count the number of relevant items in the recommendations
        relevant_and_recommended = set(relevant_items).intersection(set(recommended_items))
        
        return len(relevant_and_recommended) / len(relevant_items)
    
    @staticmethod
    def ndcg_at_k(recommended_items, relevant_items, relevance_scores=None, k=5):
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) at K.
        
        Args:
            recommended_items (list): List of recommended item IDs
            relevant_items (list): List of relevant item IDs
            relevance_scores (dict, optional): Dictionary mapping item IDs to relevance scores
            k (int): Number of recommendations to consider
            
        Returns:
            float: NDCG@K value
        """
        if len(recommended_items) == 0 or len(relevant_items) == 0:
            return 0.0
            
        # Consider only the top-k recommendations
        recommended_items = recommended_items[:k]
        
        # If relevance scores are not provided, use binary relevance
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant_items}
            
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_items):
            if item in relevance_scores:
                # Use 0-based indexing for i, but log base 2 of (i+2) for the position discount
                dcg += relevance_scores[item] / np.log2(i + 2)
                
        # Calculate ideal DCG (IDCG)
        ideal_items = sorted(relevant_items, key=lambda x: -relevance_scores.get(x, 0))
        idcg = 0.0
        for i, item in enumerate(ideal_items[:k]):
            idcg += relevance_scores[item] / np.log2(i + 2)
            
        # Avoid division by zero
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    @staticmethod
    def average_precision(recommended_items, relevant_items):
        """
        Calculate Average Precision.
        
        Args:
            recommended_items (list): List of recommended item IDs
            relevant_items (list): List of relevant item IDs
            
        Returns:
            float: Average Precision value
        """
        if len(recommended_items) == 0 or len(relevant_items) == 0:
            return 0.0
            
        hits = 0
        sum_precisions = 0.0
        
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i
                
        return sum_precisions / len(relevant_items) if len(relevant_items) > 0 else 0.0
    
    @staticmethod
    def diversity(recommended_items, item_features):
        """
        Calculate diversity of recommendations based on item features.
        
        Args:
            recommended_items (list): List of recommended item IDs
            item_features (DataFrame): DataFrame containing item features
            
        Returns:
            float: Diversity score (average pairwise distance)
        """
        if len(recommended_items) <= 1 or item_features is None:
            return 0.0
            
        # Get feature vectors for recommended items
        item_vectors = []
        for item in recommended_items:
            if item in item_features.index:
                item_vectors.append(item_features.loc[item].values)
                
        if len(item_vectors) <= 1:
            return 0.0
            
        # Calculate pairwise distances
        n = len(item_vectors)
        total_distance = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                # Use Euclidean distance
                distance = np.sqrt(np.sum((item_vectors[i] - item_vectors[j])**2))
                total_distance += distance
                count += 1
                
        # Return average distance
        return total_distance / count if count > 0 else 0.0
    
    @staticmethod
    def coverage(recommended_items_per_user, all_items):
        """
        Calculate catalog coverage of recommendations.
        
        Args:
            recommended_items_per_user (list): List of lists, where each inner list contains
                                              recommended items for a user
            all_items (list): List of all available items
            
        Returns:
            float: Coverage ratio (0 to 1)
        """
        if not recommended_items_per_user or not all_items:
            return 0.0
            
        # Flatten the list of recommended items
        all_recommended = set()
        for items in recommended_items_per_user:
            all_recommended.update(items)
            
        # Calculate coverage
        return len(all_recommended) / len(all_items)