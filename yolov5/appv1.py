from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess


app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results/exp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER


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
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
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
        
        if  image_uploaded == False:
            #Command to clasify image using the trained YOLOv5 model
            cmd_command = f"python classify/predict.py --weights {training_weights} --source {image_path}"  
            # Execute the command using subprocess
            process = subprocess.Popen(cmd_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Wait for the process to finish
            process.wait()
            
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