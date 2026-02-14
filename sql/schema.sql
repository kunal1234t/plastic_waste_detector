DROP TABLE IF EXISTS refresh_tokens;
DROP TABLE IF EXISTS actions;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS zone_metrics;
DROP TABLE IF EXISTS detections;
DROP TABLE IF EXISTS zones;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  role ENUM('admin','viewer') DEFAULT 'viewer',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE zones (
  id VARCHAR(20) PRIMARY KEY,
  name VARCHAR(255),
  latitude DECIMAL(9,6),
  longitude DECIMAL(9,6),
  description TEXT
);

CREATE TABLE detections (
  id INT AUTO_INCREMENT PRIMARY KEY,
  zone_id VARCHAR(20),
  plastic_type VARCHAR(50),
  confidence DECIMAL(4,2),
  detected_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (zone_id) REFERENCES zones(id) ON DELETE CASCADE
);

CREATE TABLE zone_metrics (
  id INT AUTO_INCREMENT PRIMARY KEY,
  zone_id VARCHAR(20),
  total_detections INT DEFAULT 0,
  risk_score INT DEFAULT 0,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (zone_id) REFERENCES zones(id) ON DELETE CASCADE
);

CREATE TABLE predictions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  zone_id VARCHAR(20),
  prediction_window VARCHAR(20),
  expected_risk INT,
  confidence DECIMAL(4,2),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (zone_id) REFERENCES zones(id) ON DELETE CASCADE
);

CREATE TABLE actions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  zone_id VARCHAR(20),
  action_type VARCHAR(50),
  status ENUM('pending','completed') DEFAULT 'pending',
  dispatched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP NULL,
  FOREIGN KEY (zone_id) REFERENCES zones(id) ON DELETE CASCADE
);

CREATE TABLE refresh_tokens (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT,
  token TEXT,
  expires_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
