import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

class CollaborativeFiltering:
    """
    Collaborative filtering recommendation system using user-based and item-based approaches.
    """
    
    def __init__(self, user_item_matrix=None):
        """
        Initialize the collaborative filtering model.
        
        Args:
            user_item_matrix (DataFrame): User-item matrix where rows are users,
                                         columns are items, and values are ratings
        """
        self.user_item_matrix = user_item_matrix
        self.user_similarity = None
        self.item_similarity = None
        self.user_factors = None
        self.item_factors = None
        self.mean_ratings = None
        
    def fit(self, user_item_matrix=None):
        """
        Fit the collaborative filtering model to the user-item matrix.
        
        Args:
            user_item_matrix (DataFrame, optional): User-item matrix
            
        Returns:
            self: The fitted model
        """
        if user_item_matrix is not None:
            self.user_item_matrix = user_item_matrix
            
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not provided")
            
        # Calculate user and item similarity matrices
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        return self
    
    def fit_svd(self, user_item_matrix=None, n_factors=20):
        """
        Fit the SVD model for matrix factorization.
        
        Args:
            user_item_matrix (DataFrame, optional): User-item matrix
            n_factors (int): Number of latent factors
            
        Returns:
            self: The fitted model
        """
        if user_item_matrix is not None:
            self.user_item_matrix = user_item_matrix
            
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not provided")
            
        # Convert to numpy array
        matrix = self.user_item_matrix.values
        
        # Calculate the mean rating for each user
        self.mean_ratings = np.mean(matrix, axis=1)
        
        # Determine the maximum number of factors based on matrix dimensions
        min_dimension = min(matrix.shape)
        if min_dimension <= 1:
            print("Warning: Matrix is too small for SVD. Using default values.")
            self.user_factors = np.zeros((matrix.shape[0], 1))
            self.item_factors = np.zeros((matrix.shape[1], 1))
            return self
            
        # Adjust n_factors if it's too large for the matrix
        if n_factors >= min_dimension:
            n_factors = min_dimension - 1
            print(f"Warning: Reducing factors to {n_factors} due to matrix dimensions {matrix.shape}")
        
        # Center the ratings matrix
        matrix_centered = matrix - self.mean_ratings.reshape(-1, 1)
        
        # Perform SVD
        u, sigma, vt = svds(matrix_centered, k=n_factors)
        
        # Convert sigma to diagonal matrix
        sigma_diag = np.diag(sigma)
        
        # Store the factors
        self.user_factors = u
        self.item_factors = vt.T
        
        return self
    
    def recommend_user_based(self, user_id, n_recommendations=5, min_similarity=0):
        """
        Generate recommendations using user-based collaborative filtering.
        
        Args:
            user_id: ID of the user
            n_recommendations (int): Number of recommendations to generate
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            DataFrame: Recommended items with predicted ratings
        """
        if self.user_similarity is None:
            self.fit()
            
        # Get the user's index in the matrix
        if isinstance(user_id, int) and user_id in self.user_item_matrix.index:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
        else:
            raise ValueError(f"User ID {user_id} not found in the user-item matrix")
            
        # Get the user's ratings
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        
        # Get the user's similarity scores with other users
        user_sim_scores = self.user_similarity[user_idx]
        
        # Filter out users with similarity below threshold
        similar_users = np.where(user_sim_scores >= min_similarity)[0]
        
        # Remove the user itself
        similar_users = similar_users[similar_users != user_idx]
        
        # Get the similarity scores of similar users
        sim_scores = user_sim_scores[similar_users]
        
        # Get the ratings of similar users
        similar_user_ratings = self.user_item_matrix.iloc[similar_users].values
        
        # Calculate weighted ratings
        weighted_ratings = sim_scores.reshape(-1, 1) * similar_user_ratings
        
        # Calculate the sum of similarity scores for each item
        sim_sums = np.sum(np.abs(sim_scores.reshape(-1, 1) * (similar_user_ratings > 0)), axis=0)
        sim_sums[sim_sums == 0] = 1  # Avoid division by zero
        
        # Calculate predicted ratings
        predicted_ratings = np.sum(weighted_ratings, axis=0) / sim_sums
        
        # Create a DataFrame with predicted ratings
        recommendations = pd.DataFrame({
            'item_id': self.user_item_matrix.columns,
            'predicted_rating': predicted_ratings
        })
        
        # Filter out items the user has already rated
        rated_items = self.user_item_matrix.columns[user_ratings > 0]
        recommendations = recommendations[~recommendations['item_id'].isin(rated_items)]
        
        # Sort by predicted rating and take top n
        recommendations = recommendations.sort_values('predicted_rating', ascending=False).head(n_recommendations)
        
        return recommendations
    
    def recommend_item_based(self, user_id, n_recommendations=5, min_similarity=0):
        """
        Generate recommendations using item-based collaborative filtering.
        
        Args:
            user_id: ID of the user
            n_recommendations (int): Number of recommendations to generate
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            DataFrame: Recommended items with predicted ratings
        """
        if self.item_similarity is None:
            self.fit()
            
        # Get the user's index in the matrix
        if isinstance(user_id, int) and user_id in self.user_item_matrix.index:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
        else:
            raise ValueError(f"User ID {user_id} not found in the user-item matrix")
            
        # Get the user's ratings
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        
        # Calculate predicted ratings for all items
        predicted_ratings = np.zeros(len(self.user_item_matrix.columns))
        
        # Get indices of items the user has rated
        rated_indices = np.where(user_ratings > 0)[0]
        
        # For each item the user hasn't rated
        for i in range(len(self.user_item_matrix.columns)):
            if user_ratings[i] > 0:
                continue  # Skip items the user has already rated
                
            # Get similarity scores between this item and all items the user has rated
            item_sim_scores = self.item_similarity[i, rated_indices]
            
            # Filter out items with similarity below threshold
            similar_items = np.where(item_sim_scores >= min_similarity)[0]
            
            if len(similar_items) == 0:
                continue
                
            # Get the similarity scores of similar items
            sim_scores = item_sim_scores[similar_items]
            
            # Get the user's ratings for similar items
            similar_item_ratings = user_ratings[rated_indices[similar_items]]
            
            # Calculate weighted rating
            weighted_sum = np.sum(sim_scores * similar_item_ratings)
            sim_sum = np.sum(np.abs(sim_scores))
            
            if sim_sum > 0:
                predicted_ratings[i] = weighted_sum / sim_sum
        
        # Create a DataFrame with predicted ratings
        recommendations = pd.DataFrame({
            'item_id': self.user_item_matrix.columns,
            'predicted_rating': predicted_ratings
        })
        
        # Filter out items the user has already rated
        rated_items = self.user_item_matrix.columns[user_ratings > 0]
        recommendations = recommendations[~recommendations['item_id'].isin(rated_items)]
        
        # Sort by predicted rating and take top n
        recommendations = recommendations.sort_values('predicted_rating', ascending=False).head(n_recommendations)
        
        return recommendations
    
    def recommend_svd(self, user_id, n_recommendations=5):
        """
        Generate recommendations using SVD matrix factorization.
        
        Args:
            user_id: ID of the user
            n_recommendations (int): Number of recommendations to generate
            
        Returns:
            DataFrame: Recommended items with predicted ratings
        """
        if self.user_factors is None or self.item_factors is None:
            self.fit_svd()
            
        # Get the user's index in the matrix
        if isinstance(user_id, int) and user_id in self.user_item_matrix.index:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
        else:
            raise ValueError(f"User ID {user_id} not found in the user-item matrix")
            
        # Get the user's ratings
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        
        # Calculate predicted ratings
        user_bias = self.mean_ratings[user_idx]
        user_factor = self.user_factors[user_idx, :]
        
        predicted_ratings = user_bias + np.dot(user_factor, self.item_factors.T)
        
        # Create a DataFrame with predicted ratings
        recommendations = pd.DataFrame({
            'item_id': self.user_item_matrix.columns,
            'predicted_rating': predicted_ratings
        })
        
        # Filter out items the user has already rated
        rated_items = self.user_item_matrix.columns[user_ratings > 0]
        recommendations = recommendations[~recommendations['item_id'].isin(rated_items)]
        
        # Sort by predicted rating and take top n
        recommendations = recommendations.sort_values('predicted_rating', ascending=False).head(n_recommendations)
        
        return recommendations