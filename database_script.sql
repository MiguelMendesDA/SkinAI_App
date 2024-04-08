CREATE DATABASE skin_cancer_app;
USE skin_cancer_app;

CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    profile_photo_path BLOB
);

CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    image_data BLOB,
    prediction_result VARCHAR(50),
    anatomical_site VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    original_images BLOB,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
