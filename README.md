# 🌱 Plant AI - Smart Agriculture Disease Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PHP](https://img.shields.io/badge/PHP-7.4%2B-purple.svg)](https://www.php.net/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.0-38bdf8.svg)](https://tailwindcss.com/)

## 📖 Overview

**Plant AI** is an intelligent agricultural disease detection and crop recommendation system powered by deep learning. It empowers farmers and growers with instant, accurate plant disease diagnosis, wheat stem rust detection, and data-driven crop recommendations based on environmental conditions.

### ✨ Key Features

- 🔬 **Plant Disease Detection** - AI-powered identification of 38+ plant diseases using CNN models
- 🌾 **Wheat Stem Rust Analysis** - Specialized detection for wheat crop health monitoring  
- 🌍 **Smart Crop Recommendation** - ML-based suggestions using soil and climate data
- 📊 **Interactive Dashboard** - Real-time analysis history and visualization
- 🎨 **Modern UI/UX** - Responsive design with dark mode support
- 📤 **Export Reports** - Download analysis results as images
- 🗺️ **Map Integration** - Geographic crop recommendations

## 🏗️ Tech Stack

### Backend
- **Python 3.8+** - Core ML/DL engine
- **PyTorch** - Deep learning framework for disease detection
- **Flask/FastAPI** - RESTful API servers
- **NumPy & Pandas** - Data processing
- **PIL/OpenCV** - Image preprocessing

### Frontend
- **PHP 7.4+** - Server-side logic
- **MySQL** - Database management
- **TailwindCSS** - Utility-first CSS framework
- **Alpine.js** - Lightweight JavaScript framework
- **Material Symbols** - Icon library

### Models
- Custom CNN for plant disease classification
- Transfer learning models (ResNet/EfficientNet)
- Random Forest for crop recommendation

## 📁 Project Structure

```
chins_majorproject/
├── backend/
│   ├── plant_api.py              # Plant disease detection API
│   ├── wheat_api.py              # Wheat stem rust detection API
│   ├── crop_api.py               # Crop recommendation API
│   ├── models/
│   │   ├── plant_disease_cnn.pth # Trained plant disease model
│   │   └── wheat_stem_model_final.pth # Trained wheat model
│   └── chin-majorproject.ipynb   # Model training notebook
├── frontend/
│   ├── index.php                 # Main dashboard
│   ├── auth.php                  # Authentication & landing page
│   ├── predict_plant.php         # Plant disease prediction
│   ├── predict_wheat.php         # Wheat disease prediction
│   ├── recommend_crop.php        # Crop recommendation
│   ├── recommend_map.php         # Map-based recommendations
│   ├── config.php                # Database configuration
│   ├── style.css                 # Custom styles
│   ├── includes/
│   │   └── global_head_scripts.php # Shared scripts
│   └── results/                  # Stored prediction results
├── database/
│   └── chin_major.sql            # Database schema
└── README.md
```

## 🚀 Installation & Setup

### Prerequisites

- **XAMPP** (Apache + MySQL)
- **Python 3.8+**
- **pip** package manager
- **Git** (optional)

### Step 1: Clone Repository

```bash
git clone https://github.com/chinmayee60/chins_majorproject.git
cd chins_majorproject
```

### Step 2: Install Python Dependencies

```bash
pip install torch torchvision
pip install flask flask-cors
pip install numpy pandas pillow opencv-python
pip install scikit-learn
```

### Step 3: Database Setup

1. Open **XAMPP Control Panel**
2. Start **Apache** and **MySQL** services
3. Navigate to `http://localhost/phpmyadmin`
4. Create a new database named `chin_major`
5. Import the SQL file:
   ```
   Import → Choose File → database/chin_major.sql → Go
   ```

### Step 4: Configure Database Connection

Edit `frontend/config.php` with your database credentials:

```php
<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "chin_major";
?>
```

### Step 5: Start Backend Services

Open **3 separate terminal windows** and run:

**Terminal 1 - Plant Disease API:**
```bash
cd backend
python plant_api.py
```

**Terminal 2 - Wheat Disease API:**
```bash
cd backend
python wheat_api.py
```

**Terminal 3 - Crop Recommendation API:**
```bash
cd backend
python crop_api.py
```

### Step 6: Launch Application

1. Ensure XAMPP Apache is running
2. Copy project to XAMPP's `htdocs` folder
3. Navigate to: `http://localhost/chins_majorproject/chins_majorproject/frontend/auth.php?landing=1`

## 🎯 Usage

### 1. **Plant Disease Detection**
- Upload a clear image of the affected plant leaf
- Receive instant diagnosis with confidence scores
- Get treatment recommendations and severity assessment

### 2. **Wheat Stem Rust Analysis**
- Upload wheat crop images
- Detect stem rust disease with precision
- Access preventive measures and care instructions

### 3. **Crop Recommendation**
- Input soil parameters (N, P, K, pH)
- Provide environmental data (temperature, humidity, rainfall)
- Get top 3 suitable crop suggestions

### 4. **View Analysis History**
- Track all past predictions
- Export results as images
- Review recommendations over time

## 🔐 Authentication

**Default Test Credentials:**
- Username: `testuser`
- Password: `password123`

**Registration:**
- Create a new account via the Register tab
- All passwords are securely hashed using PHP's `password_hash()`

## 📊 API Endpoints

### Plant Disease Detection
```
POST http://localhost:5000/predict
Body: { "image": "<base64_encoded_image>" }
```

### Wheat Disease Detection
```
POST http://localhost:5001/predict
Body: { "image": "<base64_encoded_image>" }
```

### Crop Recommendation
```
POST http://localhost:5002/recommend
Body: {
  "N": 90, "P": 42, "K": 43,
  "temperature": 20.87, "humidity": 82.0,
  "ph": 6.5, "rainfall": 202.9
}
```

## 🎨 Features Showcase

- **Gradient Hero Section** with AI-powered badge animation
- **Quick Action Cards** for instant navigation
- **Enhanced Analysis History** with color-coded disease types
- **Professional Modal Viewer** with Plant AI branding
- **Bento Grid Layouts** for result visualization
- **Dark Mode Support** with seamless theme switching
- **Responsive Design** optimized for mobile, tablet, and desktop

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Chinmayee**
- GitHub: [@chinmayee60](https://github.com/chinmayee60)

## 🙏 Acknowledgments

- Plant disease dataset from PlantVillage
- PyTorch team for the deep learning framework
- TailwindCSS for the utility-first CSS framework
- Material Symbols for the icon library

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the development team.

---

**⭐ If you find this project helpful, please consider giving it a star!**
