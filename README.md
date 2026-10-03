# Pneumonia AI Clinical Portal

A Flask web application prototype for classifying chest X-ray images as **NORMAL** or **PNEUMONIA**. It combines a DenseNet-121 feature extractor with a Transformer encoder, applies CLAHE image enhancement, and generates Grad-CAM visualizations alongside each prediction.

> **Medical disclaimer:** This project is for educational and research use only. It is not a medical device and must not be used to diagnose, treat, or make decisions about patients. Predictions and Grad-CAM visualizations are not a substitute for review by a qualified healthcare professional.

## Features

- Patient, doctor, and administrator portal views
- Chest X-ray classification with a hybrid DenseNet-121 + Transformer model
- CLAHE preprocessing and Grad-CAM overlays
- SQLite storage for user profiles, reports, and audit history
- SHA-256 matching to flag repeated image uploads
- Optional training on a folder-based image dataset

## Requirements

- Python 3.10 or later
- Packages listed in [`requirements.txt`](requirements.txt)
- The model checkpoint `pneumonia_hybrid_model_new.pth` in the project root
- Internet access on first model initialization if the pretrained DenseNet-121 weights are not already cached

The included PyTorch checkpoint is required for meaningful predictions. If it is missing, the application initializes a model without the trained project weights.

## Setup and run

Run these commands from the project root. On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. The SQLite database is initialized automatically, and uploaded scans and generated heatmaps are stored under `static/uploads/`.

## Training

`train.py` expects images arranged in class-specific folders:

```text
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

Train from the project root with:

```powershell
python train.py
```

The script selects CUDA when available and saves the best validation checkpoint as `pneumonia_hybrid_model_new.pth`, the checkpoint used by the web application.

## Project layout

```text
app.py                         Flask routes, portal logic, and report storage
model.py                       Model architecture, CLAHE, and Grad-CAM
train.py                       Training pipeline
requirements.txt               Python dependencies
dataset/                       Training, validation, and test image folders
templates/                     Flask/Jinja pages
static/                        Static assets and uploaded scan outputs
clinical_portal.db             SQLite database, created on first launch
pneumonia_hybrid_model_new.pth Model checkpoint loaded by the application
```

## Deployment and data warning

This repository is a development prototype, not a production-ready clinical system. Before exposing it to users, replace the hard-coded Flask secret and demo account credentials, add secure password storage and production authentication, configure a production WSGI server, and review privacy, consent, data retention, and access-control requirements. Do not commit real patient data, generated reports, or credentials to a public repository.
