from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess
import shlex
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler

# Initialize Flask app
app = Flask(__name__)
# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results/exp'
app.config['TRAINING_WEIGHTS'] = 'runs/train-cls/exp18/weights/best.pt'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Ensure upload and results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Set up logging
if not os.path.exists('logs'):
    os.mkdir('logs')

file_handler = RotatingFileHandler('logs/flask_app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)


# Function to resize the uploaded image
def resize_image(image_path, target_size=(224, 224)):
    app.logger.info(f"Resizing image: {image_path} to size {target_size}")
    try:
        img = Image.open(image_path)
        img = img.resize(target_size, Image.LANCZOS)
        img.save(image_path)
        return True, None
    except Exception as e:
        error_msg = f"Error resizing image: {str(e)}"
        app.logger.error(error_msg)
        return False, error_msg

# Function to check if the file is an image
def is_image_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    result = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    return result

# Function to check if the image is already uploaded
def is_image_uploaded(image_name):
    result_image_names = os.listdir(app.config['RESULTS_FOLDER'])
    result = image_name in result_image_names
    return result

# Route to the index page
@app.route('/')
def index():

    return render_template('index.html')

# Route to handle image upload
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        error_msg = "No file selected for upload."
        app.logger.error(error_msg)
        return render_template('error.html', error_message=error_msg)
    
    file = request.files['file']
    if file.filename == '':
        error_msg = "No file selected for upload."
        app.logger.error(error_msg)
        return render_template('error.html', error_message=error_msg)
    
    if file and is_image_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            # Resize the uploaded image
            success, error_msg = resize_image(file_path)
            if not success:
                raise Exception(error_msg)
            return redirect(url_for('index'))
        except Exception as e:
            error_msg = f"Error uploading file: {str(e)}"
            app.logger.error(error_msg)
            return render_template('error.html', error_message=error_msg)
    else:
        error_msg = "Invalid file format. Please upload an image file."
        app.logger.error(error_msg)
        return render_template('error.html', error_message=error_msg)

# Function to classify a single image
def classify_image(image_path):
    try:
        # Command to classify image using the trained YOLOv5 model
        cmd_command = [
            "python", "classify/predict.py",
            "--weights", app.config['TRAINING_WEIGHTS'],
            "--source", image_path
        ]
        # Execute the command using subprocess
        process = subprocess.Popen(cmd_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise Exception(stderr.decode())
    except Exception as e:
        error_msg = f"Error classifying image: {str(e)}"
        app.logger.error(error_msg)
        return False, error_msg
    return True, None

# Route to classify images
@app.route('/classify', methods=['POST'])
def classify():
    image_names = os.listdir(app.config['UPLOAD_FOLDER'])
    
    for image in image_names:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image)
        
        if not is_image_uploaded(image):
            success, error_msg = classify_image(image_path)
            if not success:
                return render_template('error.html', error_message=error_msg)
    
    result_image_names = os.listdir(app.config['RESULTS_FOLDER'])
    return render_template('results.html', result_image_names=result_image_names)

# Route to open the gallery page
@app.route('/gallery')
def gallery():
    image_names = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('gallery.html', image_names=image_names)

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=8000)
