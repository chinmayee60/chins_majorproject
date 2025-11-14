-- Database: chin_major

-- Table: users (For User Authentication)
CREATE TABLE users (
    id INT(11) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: predictions (For History and Storage)
CREATE TABLE predictions (
    id INT(11) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT(11) UNSIGNED NOT NULL,
    model_type ENUM('plant', 'wheat', 'crop') NOT NULL,
    result_file VARCHAR(255) NOT NULL, -- Path to the .txt file in the /results/ folder
    input_data TEXT, -- Stores the NPK values or original image filename/details
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Insert a dummy user for testing the prediction features
-- The password hash corresponds to 'password123'
INSERT INTO users (username, email, password_hash) VALUES 
('testuser', 'test@agro.ai', '$2y$10$wKx6D/iH7X0l9R/sA5sN4.tYwG.X7Z7k/yV5YyG1W6jB9gYv8D0S');