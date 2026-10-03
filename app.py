import os
import hashlib
import sqlite3
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from model import HybridPneumoniaModel, apply_clahe, generate_gradcam
from PIL import Image
from datetime import datetime
import time
from google import genai
from google.genai import types

app = Flask(__name__)
app.secret_key = 'pneumonia_enterprise_clinical_key_v10'

# Environment Variable Key Initialization with Fallback Check
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY environment variable is missing. Chatbot features will fail.")

gemini_client = genai.Client(api_key=api_key) if api_key else None

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DB_PATH = 'clinical_portal.db'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------------
# X-Ray Validation (Handles Blue-tinted X-rays)
# ---------------------------------------------------------
def is_valid_xray(image_path_or_pil):
    try:
        if isinstance(image_path_or_pil, str):
            ext = os.path.splitext(image_path_or_pil)[1].lower()
            if ext == '.dcm':
                return True
            img_bgr = cv2.imread(image_path_or_pil)
        else:
            img_np = np.array(image_path_or_pil.convert('RGB'))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        if img_bgr is None:
            return False

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mean_sat = np.mean(saturation)
        
        b, g, r = cv2.split(img_bgr.astype(float))
        channel_std = np.std([np.mean(r), np.mean(g), np.mean(b)])

        if mean_sat > 85.0 and channel_std > 35.0:
            return False

        return True
    except Exception:
        return False

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            patient_type TEXT DEFAULT 'PUBLIC',
            attending_doctor TEXT DEFAULT 'Unassigned',
            institution TEXT NOT NULL,
            status TEXT DEFAULT 'ACTIVE'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            patient_name TEXT NOT NULL,
            patient_age INTEGER NOT NULL,
            patient_gender TEXT NOT NULL,
            clinical_history TEXT,
            evaluator_name TEXT NOT NULL,
            evaluator_role TEXT NOT NULL,
            patient_type TEXT NOT NULL,
            institution_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            original_img TEXT NOT NULL,
            heatmap_img TEXT NOT NULL,
            img_hash TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            performed_by TEXT NOT NULL,
            user_role TEXT NOT NULL,
            action TEXT NOT NULL,
            target_report_id INTEGER,
            before_state TEXT,
            after_state TEXT,
            details TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Migration Checks
    cursor.execute("PRAGMA table_info(users)")
    user_cols = [col[1] for col in cursor.fetchall()]
    if 'patient_type' not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN patient_type TEXT DEFAULT 'PUBLIC'")
    if 'attending_doctor' not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN attending_doctor TEXT DEFAULT 'Unassigned'")
    if 'status' not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'ACTIVE'")

    cursor.execute("PRAGMA table_info(reports)")
    report_cols = [col[1] for col in cursor.fetchall()]
    if 'patient_type' not in report_cols:
        cursor.execute("ALTER TABLE reports ADD COLUMN patient_type TEXT DEFAULT 'PUBLIC'")
    if 'img_hash' not in report_cols:
        cursor.execute("ALTER TABLE reports ADD COLUMN img_hash TEXT")

    cursor.execute("PRAGMA table_info(audit_logs)")
    audit_cols = [col[1] for col in cursor.fetchall()]
    if 'target_report_id' not in audit_cols:
        cursor.execute("ALTER TABLE audit_logs ADD COLUMN target_report_id INTEGER")
    if 'before_state' not in audit_cols:
        cursor.execute("ALTER TABLE audit_logs ADD COLUMN before_state TEXT")
    if 'after_state' not in audit_cols:
        cursor.execute("ALTER TABLE audit_logs ADD COLUMN after_state TEXT")

    # Seed Admin User
    cursor.execute("SELECT * FROM users WHERE LOWER(email)='admin@mail.com'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (email, password, name, role, patient_type, attending_doctor, institution, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       ('admin@mail.com', '123', 'System Administrator', 'Administrator', 'N/A', 'N/A', 'General Diagnostic Health Center', 'ACTIVE'))
    else:
        cursor.execute("UPDATE users SET role='Administrator', status='ACTIVE' WHERE LOWER(email)='admin@mail.com'")

    # Seed Default Doctors
    default_doctors = [
        ('dr.smith@hospital.org', '123', 'Dr. Smith (Pulmonology)'),
        ('dr.jones@hospital.org', '123', 'Dr. Jones (Radiology)'),
        ('dr.williams@hospital.org', '123', 'Dr. Williams (Clinical Care)')
    ]
    for email, pwd, name in default_doctors:
        cursor.execute("SELECT * FROM users WHERE LOWER(email)=?", (email,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO users (email, password, name, role, patient_type, attending_doctor, institution, status) VALUES (?, ?, ?, 'Doctor', 'N/A', 'N/A', 'General Diagnostic Health Center', 'ACTIVE')",
                           (email, pwd, name))

    conn.commit()
    conn.close()

init_db()

# Load Neural Network Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridPneumoniaModel(num_classes=2)
MODEL_WEIGHTS_PATH = 'pneumonia_hybrid_model_new.pth'

if os.path.exists(MODEL_WEIGHTS_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True), strict=False)
    except Exception:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device), strict=False)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASS_LABELS = {0: 'NORMAL', 1: 'PNEUMONIA'}

def compute_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE role='Doctor' AND status='ACTIVE'")
    active_doctors = cursor.fetchall()
    conn.close()
    
    return render_template('login.html', doctors=active_doctors)

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '').strip()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email)=?", (email,))
    user = cursor.fetchone()

    if user:
        if user['password'] != password:
            conn.close()
            return render_template('login.html', error="Authentication failed: Incorrect password.", doctors=[])
        
        if user['role'] == 'Doctor' and user['status'] == 'BLOCKED':
            conn.close()
            return render_template('login.html', error="Access Denied: Your doctor account has been deactivated by the Hospital Administrator.", doctors=[])

        session['user_id'] = user['id']
        session['email'] = user['email']
        session['user'] = user['name']
        session['role'] = user['role']
        session['patient_type'] = user['patient_type']
        session['attending_doctor'] = user['attending_doctor']
        session['institution'] = user['institution']

        cursor.execute("INSERT INTO audit_logs (performed_by, user_role, action, before_state, after_state, details) VALUES (?, ?, 'USER_LOGIN', 'LOGGED_OUT', 'LOGGED_IN', ?)",
                       (user['name'], user['role'], f"User successfully logged into portal from {email}"))
        conn.commit()
        conn.close()
        return redirect(url_for('dashboard'))
    else:
        patient_type = 'PUBLIC'
        role = 'Patient'
        
        cursor.execute("INSERT INTO users (email, password, name, role, patient_type, attending_doctor, institution, status) VALUES (?, ?, ?, ?, ?, 'N/A', ?, ?)",
                       (email, password, email.split('@')[0].capitalize(), role, patient_type, 'General Diagnostic Health Center', 'ACTIVE'))
        conn.commit()
        
        cursor.execute("SELECT * FROM users WHERE LOWER(email)=?", (email,))
        new_user = cursor.fetchone()
        
        session['user_id'] = new_user['id']
        session['email'] = new_user['email']
        session['user'] = new_user['name']
        session['role'] = new_user['role']
        session['patient_type'] = new_user['patient_type']
        session['attending_doctor'] = new_user['attending_doctor']
        session['institution'] = new_user['institution']

        cursor.execute("INSERT INTO audit_logs (performed_by, user_role, action, before_state, after_state, details) VALUES (?, ?, 'AUTO_REGISTER', 'UNREGISTERED', 'REGISTERED', ?)",
                       (new_user['name'], new_user['role'], f"New Patient dynamically registered with email {email}"))
        conn.commit()
        conn.close()
        return redirect(url_for('dashboard'))

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password')
    patient_type = request.form.get('patient_type', 'PUBLIC')
    attending_doctor = request.form.get('attending_doctor', 'Unassigned') if patient_type == 'PRIVATE_HOSPITAL' else 'N/A'
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email)=?", (email,))
    if cursor.fetchone():
        cursor.execute("SELECT name FROM users WHERE role='Doctor' AND status='ACTIVE'")
        active_doctors = cursor.fetchall()
        conn.close()
        return render_template('login.html', error="An account with this email address already exists.", doctors=active_doctors)
    
    cursor.execute("INSERT INTO users (email, password, name, role, patient_type, attending_doctor, institution, status) VALUES (?, ?, ?, 'Patient', ?, ?, 'General Diagnostic Health Center', 'ACTIVE')",
                   (email, password, name, patient_type, attending_doctor))
    conn.commit()
    conn.close()
    return render_template('login.html', success="Patient profile created successfully. You may now sign in.", doctors=[])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('index'))
    
    user_id = session['user_id']
    user_role = session['role']
    user_name = session['user']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if user_role == 'Administrator':
        cursor.execute("SELECT * FROM reports ORDER BY id DESC")
        history_records = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM reports WHERE prediction='PNEUMONIA'")
        pneumonia_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM reports WHERE prediction='NORMAL'")
        normal_count = cursor.fetchone()[0]

    elif user_role == 'Doctor':
        cursor.execute("""
            SELECT * FROM reports 
            WHERE LOWER(evaluator_name)=LOWER(?) 
               OR user_id IN (SELECT id FROM users WHERE LOWER(attending_doctor)=LOWER(?))
            ORDER BY id DESC
        """, (user_name, user_name))
        history_records = cursor.fetchall()
        
        cursor.execute("""
            SELECT COUNT(*) FROM reports 
            WHERE (LOWER(evaluator_name)=LOWER(?) OR user_id IN (SELECT id FROM users WHERE LOWER(attending_doctor)=LOWER(?)))
              AND prediction='PNEUMONIA'
        """, (user_name, user_name))
        pneumonia_count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM reports 
            WHERE (LOWER(evaluator_name)=LOWER(?) OR user_id IN (SELECT id FROM users WHERE LOWER(attending_doctor)=LOWER(?)))
              AND prediction='NORMAL'
        """, (user_name, user_name))
        normal_count = cursor.fetchone()[0]

    else:
        cursor.execute("SELECT * FROM reports WHERE user_id=? OR LOWER(patient_name)=LOWER(?) ORDER BY id DESC", (user_id, user_name))
        history_records = cursor.fetchall()
        
        cursor.execute("SELECT COUNT(*) FROM reports WHERE (user_id=? OR LOWER(patient_name)=LOWER(?)) AND prediction='PNEUMONIA'", (user_id, user_name))
        pneumonia_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM reports WHERE (user_id=? OR LOWER(patient_name)=LOWER(?)) AND prediction='NORMAL'", (user_id, user_name))
        normal_count = cursor.fetchone()[0]

    total_count = pneumonia_count + normal_count

    doctors_list = []
    audit_logs = []
    
    if user_role == 'Administrator':
        cursor.execute("SELECT * FROM users WHERE role='Doctor'")
        doctors_list = cursor.fetchall()
        cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC")
        audit_logs = cursor.fetchall()

    conn.close()
    
    return render_template('dashboard.html', 
                           user=session['user'], 
                           role=session['role'], 
                           patient_type=session.get('patient_type', 'PUBLIC'),
                           attending_doctor=session.get('attending_doctor', 'N/A'),
                           institution=session['institution'],
                           history=history_records,
                           pneumonia_count=pneumonia_count,
                           normal_count=normal_count,
                           total_count=total_count,
                           doctors_list=doctors_list,
                           audit_logs=audit_logs)

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('index'))
    
    if session['role'] == 'Patient' and session.get('patient_type') == 'PRIVATE_HOSPITAL':
        return "Access Denied: Private Hospital Patients cannot generate direct self-evaluations.", 403

    if 'file' not in request.files:
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('dashboard.html', error="Invalid File Format. Please upload PNG, JPG, JPEG, or DICOM (.dcm) files.",
                               user=session['user'], role=session['role'], patient_type=session.get('patient_type', 'PUBLIC'),
                               attending_doctor=session.get('attending_doctor', 'N/A'), institution=session['institution'],
                               history=[], pneumonia_count=0, normal_count=0, total_count=0, doctors_list=[], audit_logs=[])

    patient_name = request.form.get('patient_name', session['user'])
    patient_age = request.form.get('patient_age', 0)
    patient_gender = request.form.get('patient_gender', 'Unspecified')
    clinical_history = request.form.get('clinical_history', 'No prior clinical notes entered.')
    institution_name = request.form.get('institution_name', session.get('institution', 'General Diagnostic Health Center'))
    bypass_duplicate = request.form.get('bypass_duplicate', 'false')

    if file:
        file_bytes = file.read()
        file.seek(0)
        img_hash = compute_file_hash(file_bytes)

        if bypass_duplicate != 'true':
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reports WHERE img_hash=?", (img_hash,))
            existing_report = cursor.fetchone()
            conn.close()

            if existing_report:
                return render_template('duplicate_warning.html', 
                                       existing=existing_report,
                                       patient_name=patient_name,
                                       patient_age=patient_age,
                                       patient_gender=patient_gender,
                                       clinical_history=clinical_history,
                                       institution_name=institution_name)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(file.filename)[1].lower()
        
        temp_filename = f"temp_{timestamp_str}{ext}"
        orig_filename = f"scan_{timestamp_str}.png"
        heatmap_filename = f"heatmap_{timestamp_str}.png"
        
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], orig_filename)
        
        file.save(temp_path)

        if not is_valid_xray(temp_path):
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return render_template('dashboard.html', 
                                   error="Invalid Image Uploaded! The selected file does not appear to be a valid Chest X-ray. Please upload a genuine Chest Radiograph.",
                                   user=session['user'], role=session['role'], patient_type=session.get('patient_type', 'PUBLIC'),
                                   attending_doctor=session.get('attending_doctor', 'N/A'), institution=session['institution'],
                                   history=[], pneumonia_count=0, normal_count=0, total_count=0, doctors_list=[], audit_logs=[])

        enhanced_pil_img = apply_clahe(temp_path)
        enhanced_pil_img.save(raw_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

        img_tensor = transform(enhanced_pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
            conf_percentage = round(confidence.item() * 100, 2)
            pred_class_idx = pred_idx.item()

        prediction = CLASS_LABELS[pred_class_idx]
        
        overlay_img, _ = generate_gradcam(model, img_tensor, img_tensor)

        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        Image.fromarray(overlay_img).save(heatmap_path)

        patient_type = session.get('patient_type', 'PUBLIC')

        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM users WHERE LOWER(name)=LOWER(?)", (patient_name,))
        target_patient = cursor.fetchone()
        assigned_user_id = target_patient['id'] if target_patient else session['user_id']

        cursor.execute('''
            INSERT INTO reports (user_id, patient_name, patient_age, patient_gender, clinical_history, evaluator_name, evaluator_role, patient_type, institution_name, prediction, confidence, original_img, heatmap_img, img_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (assigned_user_id, patient_name, patient_age, patient_gender, clinical_history, session['user'], session['role'], patient_type, institution_name, prediction, conf_percentage, orig_filename, heatmap_filename, img_hash))
        report_id = cursor.lastrowid
        
        after_state_full = (f"Report ID: #{report_id} | Patient: {patient_name} (Age: {patient_age}, Gender: {patient_gender}) | "
                           f"Result: {prediction} ({conf_percentage}% Confidence) | Evaluator: {session['user']} ({session['role']}) | "
                           f"Notes: {clinical_history}")

        cursor.execute("INSERT INTO audit_logs (performed_by, user_role, action, target_report_id, before_state, after_state, details) VALUES (?, ?, 'GENERATE_REPORT', ?, 'UNPROCESSED SCAN', ?, ?)",
                       (session['user'], session['role'], report_id, after_state_full, f"Generated evaluation report #{report_id} for Patient '{patient_name}'"))
        
        conn.commit()
        conn.close()

        return redirect(url_for('view_report', report_id=report_id))

@app.route('/report/<int:report_id>')
def view_report(report_id):
    if 'user' not in session:
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if session['role'] in ['Doctor', 'Administrator']:
        cursor.execute("SELECT * FROM reports WHERE id=?", (report_id,))
    else:
        cursor.execute("SELECT * FROM reports WHERE id=? AND (user_id=? OR LOWER(patient_name)=LOWER(?))", (report_id, session['user_id'], session['user']))
        
    record = cursor.fetchone()
    conn.close()
    
    if not record:
        return "Unauthorized Access or Report Record Not Found", 403

    return render_template('report.html', report=record, user=session['user'], role=session['role'])

@app.route('/edit_report/<int:report_id>', methods=['POST'])
def edit_report(report_id):
    if 'user' not in session or session['role'] not in ['Doctor', 'Administrator']:
        return "Unauthorized Access", 403

    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    patient_gender = request.form.get('patient_gender')
    clinical_history = request.form.get('clinical_history')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE id=?", (report_id,))
    old = cursor.fetchone()

    if old:
        before_state_full = (f"Report ID: #{old['id']} | Patient: {old['patient_name']} (Age: {old['patient_age']}, Gender: {old['patient_gender']}) | "
                             f"Result: {old['prediction']} ({old['confidence']}%) | Evaluator: {old['evaluator_name']} | Notes: {old['clinical_history']}")
        
        after_state_full = (f"Report ID: #{old['id']} | Patient: {patient_name} (Age: {patient_age}, Gender: {patient_gender}) | "
                            f"Result: {old['prediction']} ({old['confidence']}%) | Evaluator: {session['user']} | Notes: {clinical_history}")

        cursor.execute('''
            UPDATE reports 
            SET patient_name=?, patient_age=?, patient_gender=?, clinical_history=? 
            WHERE id=?
        ''', (patient_name, patient_age, patient_gender, clinical_history, report_id))

        cursor.execute('''
            INSERT INTO audit_logs (performed_by, user_role, action, target_report_id, before_state, after_state, details)
            VALUES (?, ?, 'EDIT_REPORT', ?, ?, ?, ?)
        ''', (session['user'], session['role'], report_id, before_state_full, after_state_full, f"Updated demographics and notes for Report #{report_id}"))

        conn.commit()

    conn.close()
    return redirect(url_for('view_report', report_id=report_id))

@app.route('/delete_report/<int:report_id>', methods=['POST'])
def delete_report(report_id):
    if 'user' not in session or session['role'] not in ['Doctor', 'Administrator']:
        return "Unauthorized Access", 403

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE id=?", (report_id,))
    report = cursor.fetchone()

    if report:
        before_state_full = (f"Report ID: #{report['id']} | Patient: {report['patient_name']} (Age: {report['patient_age']}, Gender: {report['patient_gender']}) | "
                             f"Result: {report['prediction']} ({report['confidence']}%) | Evaluator: {report['evaluator_name']} ({report['evaluator_role']}) | "
                             f"Notes: {report['clinical_history']}")
        
        cursor.execute("DELETE FROM reports WHERE id=?", (report_id,))
        
        cursor.execute('''
            INSERT INTO audit_logs (performed_by, user_role, action, target_report_id, before_state, after_state, details)
            VALUES (?, ?, 'DELETE_REPORT', ?, ?, 'DELETED RECORD', ?)
        ''', (session['user'], session['role'], report_id, before_state_full, f"Permanently removed Report #{report['id']} from active database"))
        
        conn.commit()

    conn.close()
    return redirect(url_for('dashboard'))

@app.route('/admin/add_doctor', methods=['POST'])
def add_doctor():
    if 'user' not in session or session['role'] != 'Administrator':
        return "Unauthorized Access", 403

    name = request.form.get('name')
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email)=?", (email,))
    existing_user = cursor.fetchone()

    if existing_user:
        cursor.execute("UPDATE users SET role='Doctor', password=?, status='ACTIVE' WHERE LOWER(email)=?", (password, email))
    else:
        cursor.execute("INSERT INTO users (email, password, name, role, patient_type, attending_doctor, institution, status) VALUES (?, ?, ?, 'Doctor', 'N/A', 'N/A', 'General Diagnostic Health Center', 'ACTIVE')",
                       (email, password, name))

    log_detail = f"Authorized Doctor account for '{name}' ({email})"
    cursor.execute("INSERT INTO audit_logs (performed_by, user_role, action, before_state, after_state, details) VALUES (?, ?, 'ADD_DOCTOR', 'UNAUTHORIZED', ?, ?)",
                   (session['user'], session['role'], f"Doctor Profile: {name} ({email}) - Active", log_detail))
    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))

@app.route('/admin/toggle_doctor_status/<int:doc_id>', methods=['POST'])
def toggle_doctor_status(doc_id):
    if 'user' not in session or session['role'] != 'Administrator':
        return "Unauthorized Access", 403

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id=?", (doc_id,))
    doctor = cursor.fetchone()

    if doctor:
        new_status = 'BLOCKED' if doctor['status'] == 'ACTIVE' else 'ACTIVE'
        cursor.execute("UPDATE users SET status=? WHERE id=?", (new_status, doc_id))
        
        log_detail = f"Changed access status of Doctor '{doctor['name']}' to {new_status}"
        cursor.execute("INSERT INTO audit_logs (performed_by, user_role, action, before_state, after_state, details) VALUES (?, ?, 'TOGGLE_DOCTOR_STATUS', ?, ?, ?)",
                       (session['user'], session['role'], f"Status: {doctor['status']}", f"Status: {new_status}", log_detail))
        conn.commit()

    conn.close()
    return redirect(url_for('dashboard'))

# ---------------------------------------------------------
# Robust Chatbot API Route with Dynamic Model Fallback
# ---------------------------------------------------------


@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint connected to the floating chatbot widget in dashboard.html."""
    try:
        if not gemini_client:
            return jsonify({
                'response': "GEMINI_API_KEY environment variable is not set on the server. Please check your system settings."
            }), 500

        data = request.get_json() or {}
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'response': 'Please enter a valid message.'}), 400

        system_prompt = (
            "You are a helpful Clinical Assistant inside a Chest X-ray Pneumonia "
            "Diagnostic Portal built with DenseNet-121 and Transformer Encoders. "
            "Assist medical personnel and patients regarding diagnostic reports, "
            "Grad-CAM visual interpretations, model confidence, and system navigation."
        )

        config_obj = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3
        )

        # Retry loop for transient 503 / network errors
        models_to_try = ['gemini-3.8-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
        
        for model_name in models_to_try:
            for attempt in range(2):  # Try each model up to 2 times
                try:
                    response = gemini_client.models.generate_content(
                        model=model_name,
                        contents=user_message,
                        config=config_obj
                    )
                    return jsonify({'response': response.text})
                except Exception as err:
                    print(f"[Gemini Retry] Attempt {attempt+1} on {model_name} failed: {err}")
                    time.sleep(1)  # Brief pause before retrying

        # Fallback response if all models/attempts are busy
        return jsonify({
            'response': "The AI Clinical Assistant is currently experiencing high demand from Google services. Please try sending your query again in a few seconds."
        })

    except Exception as e:
        print(f"Gemini Chat API Failure: {repr(e)}")
        return jsonify({
            'response': "System temporarily unavailable. Please try your request again shortly."
        }), 500
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
