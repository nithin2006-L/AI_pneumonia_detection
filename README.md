# AI_pneumonia_detection
A pneumonia detection system from chest X-ray images using hybrid convolutional neural network and vision transformer. The system uses a hybrid CNN ViT architecture to classify pneumonia and normal chest X-ray images. The CNN extracts local features and ViT extracts global features from the input images.

# pneumonia
# 🫁 AI-Powered Clinical Pneumonia Diagnostic Portal

Enterprise-grade web platform for medical radiography analysis, utilizing a Hybrid Deep Learning Model for Chest X-ray pneumonia detection, Grad-CAM visual heatmaps, and rigorous input validation and role-based access controls.
---

## 🌟 Key Features

Hybrid Deep Learning Pipeline: Custom PyTorch architecture + CLAHE image enhancement for diagnostic accuracy
Explainable AI (Grad-CAM): Visual heatmaps showing exact lung opacity regions (Red=AI attention/tissue opacification, Blue=normal aeration)
Multi-Stage X-Ray Validation: 3-layer heuristic filter (Color Saturation Delta, Dynamic Range Histogram, Spatial Luminance Distribution) to reject non-radiograph uploads (e.g., color photos, selfies, flat images) before running inference
SHA-256 Duplicate Detection: Prevent duplicate medical records, offer overwrite options
Role-Based Access Control (RBAC):
Administrator: System oversight, doctor account management/blocking, immutable audit logging
Doctor: Diagnostic evaluation, patient report management/edit/delete
Patient (Public vs. Private): View personal history and direct diagnostic self-evaluations (Public mode)
Comprehensive Audit Trail: Logs all user actions, authentication attempts, report modifications, deletions with Before & After state snapshots
Clinical Assistant Chatbot: Embedded API assistant for questions about Grad-CAM, role capabilities, portal usage
---

## 🏗️ Tech Stack

Backend Framework: Python 3.x, Flask
Deep Learning & Vision: PyTorch, Torchvision, OpenCV, PIL (Pillow), NumPy
Database: SQLite3
Frontend: HTML5, CSS3, JavaScript, Jinja2 Templates
---

🚀 Getting Started1. Prerequisites: Python 3.8 or higher installed2. Clone the Repository:Bashgit clone [https://github.com/your-username/pneumonia-diagnostic-portal.git](https://github.com/your-username/pneumonia-diagnostic-portal.git)
cd pneumonia-diagnostic-portal
3. Create a Virtual Environment & Install Dependencies:Bash# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install flask torch torchvision pillow numpy opencv-python
4. Model Weights Setup: Ensure your trained model weights file (pneumonia_hybrid_model.pth) is placed in the root directory, alongside accounts for quick testing:RoleEmailPasswordAdministratoradmin@mail.com123Doctordr.smith@hospital.org123Doctordr.jones@hospital.org123PatientRegister via UI or login with any new emailSelf-created🛡️ Input Validation System: Portal automatically screens incoming files with validate_chest_xray() in app.py:Color Saturation Filter: Calculates RGB channel variances ($\Delta_{RG}, \Delta_{GB} > 10.0$) to reject full-color photographs.Contrast Standard Deviation: Rejects solid, blank, or text-heavy low-variance images ($\sigma < 20.0$).Anatomical Spatial Luminance Check: Ensures center lung fields display brighter relative luminance compared to outer border regions.📝 Disclaimer: This portal is intended as a clinical decision support tool and educational demonstration, not a replacement for professional diagnostic judgment by a qualified radiologist or medical practitioner.

## 📁 Project Structure

```text
.
├── app.py           # Core Flask application, routes, validation logic
├── model.py          # PyTorch HybridPneumoniaModel, CLAHE, Grad-CAM routines
├── pneumonia_hybrid_model.pth # Trained PyTorch model weights file
├── clinical_portal.db     # SQLite database (auto-generated on initial launch)
├── static/
│  └── uploads/        # Saved original scans & generated heatmap overlays
└── templates/
├── login.html       # Authentication & registration interface
├── dashboard.html     # Dynamic portal dashboard by user role
├── report.html       # Detailed diagnostic report & Grad-CAM visualizer
└── duplicate_warning.html # Handling duplicate SHA-256 scan detection
```text
