document.addEventListener('DOMContentLoaded', function() {
    // User selection
    const userSelect = document.getElementById('user-select');
    userSelect.addEventListener('change', function() {
        const userId = this.value;
        if (userId) {
            loadUserProfile(userId);
            loadCollaborativeRecommendations(userId);
            loadContentBasedRecommendations(userId);
        } else {
            clearRecommendations();
        }
    });

    // Similar items buttons
    document.querySelectorAll('.similar-btn').forEach(button => {
        button.addEventListener('click', function() {
            const itemId = this.getAttribute('data-id');
            loadSimilarItems(itemId);
        });
    });
    
    // Image upload functionality
    initializeImageUpload();
});

function loadUserProfile(userId) {
    fetch(`/user/${userId}`)
        .then(response => response.json())
        .then(data => {
            const userRatings = document.getElementById('user-ratings');
            userRatings.innerHTML = '';
            
            if (data.rated_items && data.rated_items.length > 0) {
                data.rated_items.forEach(item => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    li.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <strong>${item.title}</strong>
                                <div class="text-muted">${item.category}</div>
                            </div>
                            <div>
                                ${getRatingStars(item.rating)}
                            </div>
                        </div>
                    `;
                    userRatings.appendChild(li);
                });
            } else {
                userRatings.innerHTML = '<li class="list-group-item">No ratings found</li>';
            }
            
            document.getElementById('user-profile-card').style.display = 'block';
        })
        .catch(error => {
            console.error('Error loading user profile:', error);
        });
}

function loadCollaborativeRecommendations(userId) {
    fetch(`/recommend/collaborative/${userId}`)
        .then(response => response.json())
        .then(data => {
            // User-based recommendations
            const userBasedContainer = document.getElementById('user-based-recommendations');
            userBasedContainer.innerHTML = '';
            
            if (data.user_based && data.user_based.length > 0) {
                data.user_based.forEach(item => {
                    userBasedContainer.appendChild(createRecommendationElement(item));
                });
            } else {
                userBasedContainer.innerHTML = '<div class="text-center text-muted">No recommendations found</div>';
            }
            
            // Item-based recommendations
            const itemBasedContainer = document.getElementById('item-based-recommendations');
            itemBasedContainer.innerHTML = '';
            
            if (data.item_based && data.item_based.length > 0) {
                data.item_based.forEach(item => {
                    itemBasedContainer.appendChild(createRecommendationElement(item));
                });
            } else {
                itemBasedContainer.innerHTML = '<div class="text-center text-muted">No recommendations found</div>';
            }
            
            // SVD-based recommendations
            const svdBasedContainer = document.getElementById('svd-based-recommendations');
            svdBasedContainer.innerHTML = '';
            
            if (data.svd_based && data.svd_based.length > 0) {
                data.svd_based.forEach(item => {
                    svdBasedContainer.appendChild(createRecommendationElement(item));
                });
            } else {
                svdBasedContainer.innerHTML = '<div class="text-center text-muted">No recommendations found</div>';
            }
        })
        .catch(error => {
            console.error('Error loading collaborative recommendations:', error);
        });
}

function loadContentBasedRecommendations(userId) {
    fetch(`/recommend/content/${userId}`)
        .then(response => response.json())
        .then(data => {
            const contentBasedContainer = document.getElementById('content-based-recommendations');
            contentBasedContainer.innerHTML = '';
            
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(item => {
                    contentBasedContainer.appendChild(createRecommendationElement(item));
                });
            } else {
                contentBasedContainer.innerHTML = '<div class="text-center text-muted">No recommendations found</div>';
            }
        })
        .catch(error => {
            console.error('Error loading content-based recommendations:', error);
        });
}

function loadSimilarItems(itemId) {
    fetch(`/similar/${itemId}`)
        .then(response => response.json())
        .then(data => {
            const similarItemsContainer = document.getElementById('similar-items-container');
            similarItemsContainer.innerHTML = '';
            
            if (data.similar_items && data.similar_items.length > 0) {
                data.similar_items.forEach(item => {
                    const col = document.createElement('div');
                    col.className = 'col-md-4 mb-3';
                    
                    const similarItem = document.createElement('div');
                    similarItem.className = 'similar-item';
                    similarItem.innerHTML = `
                        <h5>${item.title}</h5>
                        <div class="category">${item.category}</div>
                        <div class="similarity">Similarity: ${(item.similarity * 100).toFixed(1)}%</div>
                    `;
                    
                    col.appendChild(similarItem);
                    similarItemsContainer.appendChild(col);
                });
            } else {
                similarItemsContainer.innerHTML = '<div class="col-12 text-center text-muted">No similar items found</div>';
            }
            
            const modal = new bootstrap.Modal(document.getElementById('similar-items-modal'));
            modal.show();
        })
        .catch(error => {
            console.error('Error loading similar items:', error);
        });
}

function createRecommendationElement(item) {
    const div = document.createElement('div');
    div.className = 'recommendation-item';
    div.setAttribute('data-item-id', item.item_id);
    
    let imagePath = '';
     const title = item.title.toLowerCase();
     if (title.includes('earth') || title.includes('globe') || title.includes('world')) {
         imagePath = '/static/images/earth.svg';
     } else if (title.includes('smartphone')) {
         imagePath = '/static/images/smartphone_x.svg';
     } else if (title.includes('running shoes')) {
         imagePath = '/static/images/running_shoes.svg';
     } else if (title.includes('coffee maker')) {
         imagePath = '/static/images/coffee_maker.svg';
     } else if (title.includes('wireless headphones')) {
         imagePath = '/static/images/wireless_headphones.svg';
     } else if (title.includes('yoga mat')) {
         imagePath = '/static/images/yoga_mat.svg';
     } else if (title.includes('blender')) {
         imagePath = '/static/images/blender.svg';
     } else if (title.includes('smart watch')) {
         imagePath = '/static/images/smart_watch.svg';
     } else {
         imagePath = `/static/images/${item.category.toLowerCase()}.svg`;
     }
    
    div.innerHTML = `
        <img src="${imagePath}" class="recommendation-image" alt="${item.title}">
        <h5>${item.title}</h5>
        <div class="category">${item.category}</div>
        <div class="d-flex justify-content-between align-items-center">
            <div class="price">₹${(item.price * 83.5).toFixed(2)}</div>
            <div class="score">Score: ${item.score.toFixed(2)}</div>
        </div>
    `;
    return div;
}

function getRatingStars(rating) {
    const fullStars = Math.floor(rating);
    const halfStar = rating % 1 >= 0.5;
    const emptyStars = 5 - fullStars - (halfStar ? 1 : 0);
    
    let stars = '';
    
    // Full stars
    for (let i = 0; i < fullStars; i++) {
        stars += '<span class="rating-star">★</span>';
    }
    
    // Half star
    if (halfStar) {
        stars += '<span class="rating-star">½</span>';
    }
    
    // Empty stars
    for (let i = 0; i < emptyStars; i++) {
        stars += '<span class="rating-star" style="color: #ddd;">★</span>';
    }
    
    return stars;
}

function clearRecommendations() {
    document.getElementById('user-profile-card').style.display = 'none';
    document.getElementById('user-ratings').innerHTML = '';
    
    const placeholderText = '<div class="placeholder-text">Select a user to see recommendations</div>';
    document.getElementById('user-based-recommendations').innerHTML = placeholderText;
    document.getElementById('item-based-recommendations').innerHTML = placeholderText;
    document.getElementById('svd-based-recommendations').innerHTML = placeholderText;
    document.getElementById('content-based-recommendations').innerHTML = placeholderText;
}

// Image upload functionality
function initializeImageUpload() {
    // Setup upload button click handlers
    document.querySelectorAll('.upload-btn').forEach(button => {
        button.addEventListener('click', function() {
            const productId = this.getAttribute('data-id');
            document.getElementById(`file-input-${productId}`).click();
        });
    });

    // Setup file input change handlers
    document.querySelectorAll('.file-input').forEach(input => {
        input.addEventListener('change', function(e) {
            if (this.files && this.files[0]) {
                const productId = this.getAttribute('data-product-id');
                handleFileUpload(this.files[0], productId);
            }
        });
    });

    // Setup drag and drop handlers
    document.querySelectorAll('.upload-overlay').forEach(overlay => {
        const productId = overlay.id.replace('upload-overlay-', '');
        
        overlay.addEventListener('click', function() {
            document.getElementById(`file-input-${productId}`).click();
        });

        overlay.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('drag-over');
        });

        overlay.addEventListener('dragleave', function() {
            this.classList.remove('drag-over');
        });

        overlay.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('drag-over');
            
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFileUpload(e.dataTransfer.files[0], productId);
            }
        });
    });
}

function handleFileUpload(file, productId) {
    // Validate file type
    if (!file.type.match('image.*')) {
        alert('Please select an image file');
        return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
        alert('File size should be less than 5MB');
        return;
    }

    // Show loading state
    const imageElement = document.getElementById(`product-image-${productId}`);
    imageElement.style.opacity = '0.5';
    
    // Create progress element
    let progressElement = document.querySelector(`#upload-overlay-${productId} .upload-progress`);
    if (!progressElement) {
        progressElement = document.createElement('div');
        progressElement.className = 'upload-progress';
        document.getElementById(`upload-overlay-${productId}`).appendChild(progressElement);
    }
    
    // Create FormData
    const formData = new FormData();
    formData.append('image', file);
    formData.append('product_id', productId);
    
    // Create and configure XHR
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload-image', true);
    
    // Setup progress tracking
    xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            progressElement.style.width = percentComplete + '%';
        }
    };
    
    // Handle response
    xhr.onload = function() {
        if (xhr.status === 200) {
            const response = JSON.parse(xhr.responseText);
            
            // Show preview
            if (response.success) {
                // Create a live preview with the uploaded image
                const reader = new FileReader();
                reader.onload = function(e) {
                    imageElement.src = e.target.result;
                    imageElement.style.opacity = '1';
                    
                    // Update all instances of this product image in recommendations
                    updateProductImages(productId, e.target.result);
                };
                reader.readAsDataURL(file);
                
                // Remove progress bar after a delay
                setTimeout(() => {
                    progressElement.style.width = '0';
                }, 1000);
            } else {
                alert('Error uploading image: ' + response.message);
                imageElement.style.opacity = '1';
                progressElement.style.width = '0';
            }
        } else {
            alert('Error uploading image. Please try again.');
            imageElement.style.opacity = '1';
            progressElement.style.width = '0';
        }
    };
    
    // Handle errors
    xhr.onerror = function() {
        alert('Error uploading image. Please try again.');
        imageElement.style.opacity = '1';
        progressElement.style.width = '0';
    };
    
    // Send the request
    xhr.send(formData);
}

function updateProductImages(productId, imageUrl) {
    // Update all instances of this product image in recommendations
    document.querySelectorAll(`.recommendation-item[data-id="${productId}"] img`).forEach(img => {
        img.src = imageUrl;
    });
    
    // Update similar items if visible
    document.querySelectorAll(`.similar-item[data-id="${productId}"] img`).forEach(img => {
        img.src = imageUrl;
    });
}