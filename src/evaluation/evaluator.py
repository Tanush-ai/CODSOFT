import pandas as pd
import numpy as np
from .metrics import RecommendationMetrics

class RecommendationEvaluator:
    """
    Class for evaluating recommendation models using various metrics.
    """
    
    def __init__(self, train_data=None, test_data=None):
        """
        Initialize the evaluator with training and testing data.
        
        Args:
            train_data (DataFrame): Training data with user_id, item_id, and rating columns
            test_data (DataFrame): Testing data with user_id, item_id, and rating columns
        """
        self.train_data = train_data
        self.test_data = test_data
        self.metrics = RecommendationMetrics()
        
    def evaluate_rating_prediction(self, model, k=5):
        """
        Evaluate a model's rating prediction performance.
        
        Args:
            model: Recommendation model with a predict method
            k (int): Number of recommendations to consider
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.test_data is None:
            raise ValueError("Test data not provided")
            
        # Get actual ratings from test data
        actual_ratings = self.test_data['rating'].values
        
        # Get predicted ratings
        predicted_ratings = []
        for _, row in self.test_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            pred = model.predict(user_id, item_id)
            predicted_ratings.append(pred)
            
        # Calculate metrics
        results = {
            'RMSE': self.metrics.rmse(actual_ratings, predicted_ratings),
            'MAE': self.metrics.mae(actual_ratings, predicted_ratings)
        }
        
        return results
    
    def evaluate_ranking(self, model, k=5, relevance_threshold=3.5):
        """
        Evaluate a model's ranking performance.
        
        Args:
            model: Recommendation model with a recommend method
            k (int): Number of recommendations to consider
            relevance_threshold (float): Minimum rating to consider an item relevant
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Training or test data not provided")
            
        # Group test data by user
        test_by_user = self.test_data.groupby('user_id')
        
        # Initialize metrics
        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        ap_sum = 0.0
        user_count = 0
        
        all_recommended_items = []
        
        # Evaluate for each user
        for user_id, group in test_by_user:
            # Get relevant items for this user (items with rating >= threshold)
            relevant_items = group[group['rating'] >= relevance_threshold]['item_id'].tolist()
            
            if not relevant_items:
                continue
                
            # Get relevance scores for NDCG calculation
            relevance_scores = {row['item_id']: row['rating'] 
                               for _, row in group.iterrows()}
            
            # Get recommendations for this user
            try:
                recommendations = model.recommend(user_id, n_recommendations=k)
                recommended_items = recommendations['item_id'].tolist()
                all_recommended_items.append(recommended_items)
                
                # Calculate metrics
                precision = self.metrics.precision_at_k(recommended_items, relevant_items, k)
                recall = self.metrics.recall_at_k(recommended_items, relevant_items, k)
                ndcg = self.metrics.ndcg_at_k(recommended_items, relevant_items, relevance_scores, k)
                ap = self.metrics.average_precision(recommended_items, relevant_items)
                
                precision_sum += precision
                recall_sum += recall
                ndcg_sum += ndcg
                ap_sum += ap
                user_count += 1
                
            except Exception as e:
                print(f"Error evaluating user {user_id}: {str(e)}")
                continue
        
        # Calculate average metrics
        results = {}
        if user_count > 0:
            results = {
                f'Precision@{k}': precision_sum / user_count,
                f'Recall@{k}': recall_sum / user_count,
                f'NDCG@{k}': ndcg_sum / user_count,
                'MAP': ap_sum / user_count
            }
            
            # Calculate coverage if we have all items
            if hasattr(model, 'user_item_matrix'):
                all_items = model.user_item_matrix.columns.tolist()
                results['Coverage'] = self.metrics.coverage(all_recommended_items, all_items)
        
        return results
    
    def evaluate_diversity(self, model, item_features, k=5):
        """
        Evaluate diversity of recommendations.
        
        Args:
            model: Recommendation model with a recommend method
            item_features (DataFrame): DataFrame containing item features
            k (int): Number of recommendations to consider
            
        Returns:
            float: Average diversity score
        """
        if self.train_data is None:
            raise ValueError("Training data not provided")
            
        # Get unique users
        users = self.train_data['user_id'].unique()
        
        # Initialize diversity sum
        diversity_sum = 0.0
        user_count = 0
        
        # Evaluate for each user
        for user_id in users:
            try:
                # Get recommendations for this user
                recommendations = model.recommend(user_id, n_recommendations=k)
                recommended_items = recommendations['item_id'].tolist()
                
                # Calculate diversity
                diversity = self.metrics.diversity(recommended_items, item_features)
                diversity_sum += diversity
                user_count += 1
                
            except Exception as e:
                print(f"Error evaluating diversity for user {user_id}: {str(e)}")
                continue
        
        # Calculate average diversity
        return diversity_sum / user_count if user_count > 0 else 0.0