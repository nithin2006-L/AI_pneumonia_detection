import os
import hashlib
import sqlite3
import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from model import HybridPneumoniaModel, apply_clahe, generate_gradcam
from PIL import Image
from datetime import datetime
from flask import jsonify, session
from authlib.integrations.flask_client import OAuth
from xray_validation import validate_chest_xray

app = Flask(__name__)
app.secret_key = os.environ.get(
    'FLASK_SECRET_KEY',
    'pneumonia_enterprise_clinical_key_v10'
)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB upload limit

# -----------------------------------------------------------------------------
# Google OAuth configuration
# Keep credentials in environment variables instead of hard-coding them.
# -----------------------------------------------------------------------------
oauth = OAuth(app)

GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', 'YOUR_GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', 'YOUR_GOOGLE_CLIENT_SECRET')

google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

GOOGLE_CONFIGURED = (
    bool(GOOGLE_CLIENT_ID)
    and bool(GOOGLE_CLIENT_SECRET)
    and not GOOGLE_CLIENT_ID.startswith('YOUR_')
    and not GOOGLE_CLIENT_SECRET.startswith('YOUR_')
)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DB_PATH = 'clinical_portal.db'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

def allowed_file(filename):
    return (
        bool(filename)
        and '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def load_dicom_as_pil(dicom_path):
    """Convert a DICOM image into a normalized RGB PIL image."""
    ds = pydicom.dcmread(dicom_path, force=False)

    if not hasattr(ds, 'PixelData'):
        raise ValueError('DICOM file does not contain pixel data.')

    pixel_array = ds.pixel_array

    # The current model pipeline expects one 2-D radiograph.
    if pixel_array.ndim != 2:
        raise ValueError('Unsupported DICOM image: expected a single 2-D chest radiograph.')

    image = pixel_array.astype(np.float32)

    # Apply DICOM rescale parameters when supplied.
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    image = image * slope + intercept

    # MONOCHROME1 stores lower values as brighter pixels.
    if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
        image = np.max(image) - image

    finite_pixels = image[np.isfinite(image)]
    if finite_pixels.size == 0:
        raise ValueError('DICOM pixel data contains no valid numeric pixels.')

    # Robust 1st-99th percentile normalization.
    low, high = np.percentile(finite_pixels, [1, 99])
    if high <= low:
        low = float(np.min(finite_pixels))
        high = float(np.max(finite_pixels))

    if high <= low:
        raise ValueError('DICOM image has no usable intensity variation.')

    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    image_uint8 = (image * 255.0).astype(np.uint8)

    return Image.fromarray(image_uint8, mode='L').convert('RGB')


def validate_dicom(dicom_path):
    """Validate that a DICOM contains usable radiograph pixel data."""
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=False, force=False)

        if not hasattr(ds, 'PixelData'):
            return False, 'DICOM file does not contain pixel data.'

        pixel_array = ds.pixel_array
        if pixel_array.ndim != 2:
            return False, 'Only single-frame 2-D radiographs are supported.'

        if pixel_array.size == 0:
            return False, 'DICOM image contains empty pixel data.'

        if not np.isfinite(pixel_array.astype(np.float32)).any():
            return False, 'DICOM image contains invalid pixel values.'

        return True, ''
    except Exception as exc:
        return False, f'Invalid or unreadable DICOM file: {exc}'



def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Users Table
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
    
    # 2. Reports Table
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

    # 3. Audit Logs Table
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


# def get_stat_counts():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     try:
#         cursor.execute("SELECT COUNT(*) FROM reports WHERE prediction='PNEUMONIA'")
#         pneumonia = cursor.fetchone()[0]
#         cursor.execute("SELECT COUNT(*) FROM reports WHERE prediction='NORMAL'")
#         normal = cursor.fetchone()[0]
#     finally:
#         conn.close()
#     return pneumonia, normal

def get_stat_counts():
    conn = get_db_connection()
    cursor = conn.cursor()
    role, uid, name = session['role'], session['user_id'], session['user']
    if role == 'Administrator':
        where, params = "1=1", ()
    elif role == 'Doctor':
        where = "(LOWER(evaluator_name)=LOWER(?) OR user_id IN (SELECT id FROM users WHERE LOWER(attending_doctor)=LOWER(?)))"
        params = (name, name)
    else:
        where = "(user_id=? OR LOWER(patient_name)=LOWER(?))"
        params = (uid, name)
    cursor.execute(f"SELECT COUNT(*) FROM reports WHERE {where} AND prediction='PNEUMONIA'", params)
    pneumonia = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM reports WHERE {where} AND prediction='NORMAL'", params)
    normal = cursor.fetchone()[0]
    conn.close()
    return pneumonia, normal

# Load Model
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
    return render_dashboard()


def render_dashboard(error=None, status=200):
    """Render the dashboard with REAL data (history, counts, admin lists).
    Optionally shows a warning banner, e.g. for rejected uploads."""
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
                           error=error,
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
                           audit_logs=audit_logs), status

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
        return render_dashboard(
            error="Invalid file format. Please upload a PNG, JPG, JPEG, or DICOM (.dcm) chest X-ray.",
            status=400)

    patient_name = request.form.get('patient_name', session['user'])
    patient_age = request.form.get('patient_age', 0)
    patient_gender = request.form.get('patient_gender', 'Unspecified')
    clinical_history = request.form.get('clinical_history', 'No prior clinical notes entered.')
    institution_name = request.form.get('institution_name', session.get('institution', 'General Diagnostic Health Center'))
    bypass_duplicate = request.form.get('bypass_duplicate', 'false')

    if file:
        file_bytes = file.read()
        file.seek(0)

        # ---------------------------------------------------------
        # VALIDATION GATE: only genuine chest X-rays may continue.
        # Anything else is rejected here and never reaches the model.
        # ---------------------------------------------------------
        validation = validate_chest_xray(file_bytes, file.filename)
        if not validation.is_valid:
            conn = get_db_connection()
            conn.execute(
                "INSERT INTO audit_logs (performed_by, user_role, action, before_state, after_state, details) "
                "VALUES (?, ?, 'REJECTED_UPLOAD', 'UPLOADED', 'REJECTED', ?)",
                (session['user'], session['role'],
                 f"File '{file.filename}' rejected at stage '{validation.stage}': {validation.message}"))
            conn.commit()
            conn.close()
            return render_dashboard(
                error=f"Not a valid chest X-ray - {validation.message} "
                      f"Please upload a frontal chest radiograph (PNG, JPG or DICOM).",
                status=422)

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
        converted_dicom_path = None

        file.save(temp_path)

        try:
            # ---------------------------------------------------------
            # Check 1: Normalize the (already validated) radiograph
            # ---------------------------------------------------------
            if ext == '.dcm':
                valid_dicom, dicom_error = validate_dicom(temp_path)
                if not valid_dicom:
                    raise ValueError(dicom_error)

                dicom_pil_img = load_dicom_as_pil(temp_path)

                # Reuse the existing CLAHE implementation through a temporary PNG.
                converted_dicom_filename = f"converted_{timestamp_str}.png"
                converted_dicom_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], converted_dicom_filename
                )
                dicom_pil_img.save(converted_dicom_path, format='PNG')
                preprocessing_path = converted_dicom_path
            else:
                preprocessing_path = temp_path

            # ---------------------------------------------------------
            # Check 2: CLAHE Enhancement & Conversion to PNG
            # ---------------------------------------------------------
            enhanced_pil_img = apply_clahe(preprocessing_path)
            enhanced_pil_img = enhanced_pil_img.convert('RGB')
            enhanced_pil_img.save(raw_path, format='PNG')

        except (ValueError, pydicom.errors.InvalidDicomError) as exc:
            return render_dashboard(
                error=f"Invalid Chest X-ray/DICOM upload: {exc}",
                status=422)
        finally:
            for temporary_file in (temp_path, converted_dicom_path):
                if temporary_file and os.path.exists(temporary_file):
                    try:
                        os.remove(temporary_file)
                    except OSError:
                        pass

        img_tensor = transform(enhanced_pil_img).unsqueeze(0).to(device)
        
        # Inference
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Pneumonia class probability (Index 1)
            pneumonia_prob = probabilities[0][1].item()
            
            # Set Decision Threshold (e.g., 0.70 or 0.56)
            threshold = 0.65
            
            if pneumonia_prob >= threshold:
                pred_class_idx = 1
                confidence_val = pneumonia_prob
            else:
                pred_class_idx = 0
                confidence_val = probabilities[0][0].item() # Normal probability
                
            conf_percentage = round(confidence_val * 100, 2)

        prediction = CLASS_LABELS[pred_class_idx]
        
        # Grad-CAM Heatmap
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

@app.route('/api/chat', methods=['POST'])
def chatbot_api():
    data = request.get_json() or {}
    msg = data.get('message', '').strip().lower()

    if 'heatmap' in msg or 'color' in msg or 'red' in msg or 'blue' in msg:
        reply = "Grad-CAM Heatmaps highlight AI focus: Red/warm zones indicate dense tissue opacification or fluid accumulation, while Blue/cool zones represent normal, clear lung aeration."
    elif 'doctor' in msg or 'block' in msg or 'admin' in msg:
        reply = "Hospital Admins manage doctor rosters via the Admin Panel. Admins can register new doctors or block access if a doctor leaves the hospital."
    elif 'duplicate' in msg or 'hash' in msg:
        reply = "The system uses SHA-256 cryptographic hashing to detect identical X-rays, preventing duplicate records while giving users the option to re-evaluate if required."
    elif 'edit' in msg or 'delete' in msg or 'audit' in msg:
        reply = "Doctors and Admins can edit patient demographics or delete reports. All edit and delete operations record full Before & After states in the System Audit Log."
    elif 'private' in msg or 'public' in msg:
        reply = "Public patients can upload scans independently. Private hospital patients are linked to an attending physician, and their reports are managed directly by authorized doctors."
    else:
        reply = "Hello! I am your Clinical AI Assistant. I can answer questions about Grad-CAM heatmaps, role permissions, duplicate detection, or report auditing."

    return jsonify({'response': reply})

# -----------------------------------------------------------------------------
# Google OAuth
# -----------------------------------------------------------------------------
@app.route('/auth/google')
def google_login():
    if not GOOGLE_CONFIGURED:
        return render_template(
            'login.html',
            error='Google sign-in is not configured yet.',
            doctors=[]
        )

    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/auth/google/callback')
def google_callback():
    if not GOOGLE_CONFIGURED:
        return render_template(
            'login.html',
            error='Google sign-in is not configured yet.',
            doctors=[]
        )

    try:
        token = google.authorize_access_token()
        info = token.get('userinfo')

        if not info or not info.get('email'):
            return render_template(
                'login.html',
                error='Google sign-in did not return a valid email address.',
                doctors=[]
            )

        email = info['email'].strip().lower()
        name = info.get('name') or email.split('@')[0].capitalize()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM users WHERE LOWER(email)=?',
            (email,)
        )
        user = cursor.fetchone()

        if not user:
            cursor.execute(
                "INSERT INTO users (email, password, name, role, patient_type, attending_doctor, institution, status) "
                "VALUES (?, ?, ?, 'Patient', 'PUBLIC', 'N/A', 'General Diagnostic Health Center', 'ACTIVE')",
                (email, os.urandom(16).hex(), name)
            )
            conn.commit()
            cursor.execute(
                'SELECT * FROM users WHERE LOWER(email)=?',
                (email,)
            )
            user = cursor.fetchone()

        if not user:
            conn.close()
            return render_template(
                'login.html',
                error='Unable to create or retrieve the Google account.',
                doctors=[]
            )

        if str(user['status']).upper() != 'ACTIVE':
            conn.close()
            return render_template(
                'login.html',
                error='This account is not active. Please contact the administrator.',
                doctors=[]
            )

        conn.close()

        # Populate the same session fields used by normal login.
        session['user_id'] = user['id']
        session['email'] = user['email']
        session['user'] = user['name']
        session['role'] = user['role']
        session['patient_type'] = user['patient_type']
        session['attending_doctor'] = user['attending_doctor']
        session['institution'] = user['institution']

        return redirect(url_for('dashboard'))

    except Exception as exc:
        return render_template(
            'login.html',
            error=f'Google sign-in failed: {exc}',
            doctors=[]
        )


@app.route('/auth/facebook')
def facebook_login():
    return render_template(
        'login.html',
        error='Facebook sign-in is not configured yet.',
        doctors=[]
    )


@app.route('/api/stats')
def api_stats():
       if 'user' not in session:          # use the session key your login sets
           return jsonify(error='unauthorized'), 401

       pneumonia, normal = get_stat_counts()
       resp = jsonify(
           pneumonia_count=pneumonia,
           normal_count=normal,
           total_count=pneumonia + normal,
       )
       resp.headers['Cache-Control'] = 'no-store'
       return resp

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
