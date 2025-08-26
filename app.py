from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
import json
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from src.data_processing.data_loader import DataLoader
from src.feature_engineering.feature_extractor import FeatureExtractor
from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.content_based_filtering import ContentBasedFiltering

app = Flask(__name__, static_folder='src/ui/static', template_folder='src/ui/templates')

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'src/ui/static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize data and models
data_dir = os.path.join(os.path.dirname(__file__), 'data')
ratings_path = os.path.join(data_dir, 'sample_ratings.csv')
products_path = os.path.join(data_dir, 'sample_products.csv')

# Load data
data_loader = DataLoader(ratings_path, products_path)
ratings_data, products_data = data_loader.load_data()
user_item_matrix = data_loader.create_user_item_matrix()

# Extract features
feature_extractor = FeatureExtractor(ratings_data, products_data)
item_features = feature_extractor.extract_item_features()

# Initialize models
cf_model = CollaborativeFiltering(user_item_matrix)
cf_model.fit()
cf_model.fit_svd(n_factors=5)

cb_model = ContentBasedFiltering(item_features)
cb_model.fit()

@app.route('/')
def index():
    """Render the main page."""
    users = sorted(ratings_data['user_id'].unique())
    products = products_data.to_dict('records')
    return render_template('index.html', users=users, products=products)

@app.route('/user/<int:user_id>')
def user_profile(user_id):
    """Get user profile and ratings."""
    user_ratings = ratings_data[ratings_data['user_id'] == user_id]
    
    if user_ratings.empty:
        return jsonify({'error': 'User not found'}), 404
    
    # Get rated items with product details
    rated_items = []
    for _, row in user_ratings.iterrows():
        item_id = row['item_id']
        product = products_data[products_data['item_id'] == item_id]
        if not product.empty:
            rated_items.append({
                'item_id': item_id,
                'title': product['title'].values[0],
                'category': product['category'].values[0],
                'rating': row['rating']
            })
    
    return jsonify({
        'user_id': user_id,
        'rated_items': rated_items
    })

@app.route('/recommend/collaborative/<int:user_id>')
def recommend_collaborative(user_id):
    """Get collaborative filtering recommendations for a user."""
    try:
        # Get recommendations using different methods
        user_based = cf_model.recommend_user_based(user_id, n_recommendations=5)
        item_based = cf_model.recommend_item_based(user_id, n_recommendations=5)
        svd_based = cf_model.recommend_svd(user_id, n_recommendations=5)
        
        # Format recommendations with product details
        recommendations = {
            'user_based': format_recommendations(user_based),
            'item_based': format_recommendations(item_based),
            'svd_based': format_recommendations(svd_based)
        }
        
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recommend/content/<int:user_id>')
def recommend_content(user_id):
    """Get content-based recommendations for a user."""
    try:
        # Get user ratings
        user_ratings = ratings_data[ratings_data['user_id'] == user_id]
        
        if user_ratings.empty:
            return jsonify({'error': 'User not found'}), 404
        
        # Get recommendations
        recommendations = cb_model.recommend_for_user(user_ratings[['item_id', 'rating']], n_recommendations=5)
        
        # Format recommendations with product details
        formatted_recommendations = format_recommendations(recommendations)
        
        return jsonify({'recommendations': formatted_recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/similar/<int:item_id>')
def similar_items(item_id):
    """Get similar items based on content."""
    try:
        # Get similar items
        similar = cb_model.recommend_similar_items(item_id, n_recommendations=5)
        
        # Format recommendations with product details
        formatted_similar = []
        for _, row in similar.iterrows():
            item_id = row['item_id']
            product = products_data[products_data['item_id'] == item_id]
            if not product.empty:
                formatted_similar.append({
                    'item_id': item_id,
                    'title': product['title'].values[0],
                    'category': product['category'].values[0],
                    'similarity': float(row['similarity'])
                })
        
        return jsonify({'similar_items': formatted_similar})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def format_recommendations(recommendations):
    """Format recommendations with product details."""
    formatted = []
    for _, row in recommendations.iterrows():
        item_id = row['item_id']
        product = products_data[products_data['item_id'] == item_id]
        if not product.empty:
            formatted.append({
                'item_id': item_id,
                'title': product['title'].values[0],
                'category': product['category'].values[0],
                'price': float(product['price'].values[0]),
                'score': float(row.get('predicted_rating', row.get('predicted_score', row.get('similarity', 0))))
            })
    return formatted

# Helper functions for image upload
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def optimize_image(image_data, max_size=(800, 800), quality=85):
    """Optimize image for web display"""
    img = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if needed
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')
    
    # Resize if larger than max_size
    if img.width > max_size[0] or img.height > max_size[1]:
        img.thumbnail(max_size, Image.LANCZOS)
    
    # Save optimized image
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=quality, optimize=True)
    return output.getvalue()

@app.route('/upload-image', methods=['POST'])
def upload_image():
    """Handle image upload for products"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided'}), 400
    
    file = request.files['image']
    product_id = request.form.get('product_id')
    
    if not product_id:
        return jsonify({'success': False, 'message': 'Product ID is required'}), 400
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            filename = secure_filename(f"{product_id}_{uuid.uuid4().hex}.jpg")
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Optimize image
            optimized_image = optimize_image(file.read())
            
            # Save optimized image
            with open(file_path, 'wb') as f:
                f.write(optimized_image)
            
            # Return success response with image URL
            image_url = f"/static/uploads/{filename}"
            return jsonify({
                'success': True,
                'message': 'Image uploaded successfully',
                'image_url': image_url
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    return jsonify({'success': False, 'message': 'Invalid file type'}), 400

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)