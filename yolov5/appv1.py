from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess
import PIL
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained YOLO model weights to perform inference on the images
TRAINING_WEIGHTS = "runs/train-cls/exp18/weights/best.pt"
app.config['TRAINING_WEIGHTS'] = TRAINING_WEIGHTS

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Folder to store results
RESULTS_FOLDER = 'static/results/exp'
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

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
app.logger.info('Flask app startup')

# Function to resize the uploaded image
def resize_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size, PIL.Image.LANCZOS)
        img.save(image_path)
        return True, None
    except Exception as e:
        error_msg = f"Error resizing image: {str(e)}"
        app.logger.error(error_msg)
        return False, error_msg

# Function to check if the file is an image
def is_image_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Function to check if the image is already uploaded
def is_image_uploaded(image_name):
    result_image_names = os.listdir(app.config['RESULTS_FOLDER'])
    return image_name in result_image_names

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

# Route to classify images
@app.route('/classify', methods=['POST'])
def classify():
    image_names = os.listdir(app.config['UPLOAD_FOLDER'])
    
    for image in image_names:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image)
        
        if not is_image_uploaded(image):
            try:
                # Command to classify image using the trained YOLOv5 model
                cmd_command = f"python classify/predict.py --weights {app.config['TRAINING_WEIGHTS']} --source {image_path}"  
                # Execute the command using subprocess
                process = subprocess.Popen(cmd_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    raise Exception(stderr.decode())
            except Exception as e:
                error_msg = f"Error classifying image: {str(e)}"
                app.logger.error(error_msg)
                return render_template('error.html', error_message=error_msg)
    
    result_image_names = os.listdir(app.config['RESULTS_FOLDER'])
    return render_template('results.html', result_image_names=result_image_names)

# Route to open the gallery page
@app.route('/gallery')
def gallery():
    image_names = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('gallery.html', image_names=image_names)

if __name__ == '__main__':
    app.run(debug=True)
