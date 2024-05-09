from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess
import PIL
from PIL import Image

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results/exp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER


# Function to resize the uploaded image
def resize_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size, PIL.Image.LANCZOS)
        img.save(image_path)
        return True, None
    except Exception as e:
        error_msg = f"Error resizing image: {e}"
        return False, error_msg

def is_image_uploaded(image_name):
    # Get list of predicted images
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
        return render_template('error.html', error_message=error_msg)
    
    file = request.files['file']
    if file.filename == '':
        error_msg = "No file selected for upload."
        return render_template('error.html', error_message=error_msg)
    
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Resize the uploaded image
        success, error_msg = resize_image(file_path)
        if not success:
            # Handle error here (e.g., return an error message to the user)
            print("Error occurred during image resizing:", error_msg)
            # Redirect to some error page or display an error message to the user
            return render_template('error.html', error_message=error_msg)
        return redirect(url_for('index'))


# Route to classify images
@app.route('/classify', methods=['POST'])
def classify():
    # Implement classification logic here
    # Load the pre-trained YOLO model and perform inference on the uploaded image
    training_weights = "runs/train-cls/exp18/weights/best.pt"
    
    image_names = os.listdir(app.config['UPLOAD_FOLDER'])
    
    for image in image_names:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image)
        
        image_uploaded = is_image_uploaded(image)
        
        if not image_uploaded:
            #Command to clasify image using the trained YOLOv5 model
            cmd_command = f"python classify/predict.py --weights {training_weights} --source {image_path}"  
            # Execute the command using subprocess
            process = subprocess.Popen(cmd_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Wait for the process to finish
            #process.wait()
            
            # Capture the output and error (if any)
            error = process.communicate()
             
    # Get final list of predicted images
    result_image_names = os.listdir(app.config['RESULTS_FOLDER'])
    return render_template('results.html', result_image_names=result_image_names)
    
    #return f"Classification result will be displayed here {error}"

# Route to open the gallery page
@app.route('/gallery')
def gallery():
    # Get list of uploaded images
    image_names = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('gallery.html', image_names=image_names)

if __name__ == '__main__':
    app.run(debug=True)