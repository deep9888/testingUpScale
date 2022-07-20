from flask import Flask, render_template, jsonify, request,redirect,url_for,Response,send_from_directory,send_file
from werkzeug.utils import secure_filename
import urllib.request
from datetime import datetime
import glob
import os
import urllib 
from urllib.parse import urljoin
import requests
import torch
from PIL import Image
import numpy as np
from realesrgan import RealESRGAN
import time
import webbrowser


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4plus.pth')

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)

DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads/'
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

app.config.update(
    UPLOADED_PATH= os.path.join(basedir,'uploads'),
    DROPZONE_MAX_FILE_SIZE = 1024,
    DROPZONE_TIMEOUT = 5*60*1000)
   
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','tiff','tif'])

def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/complete')
def complete():
    '''This function redirects the user to the complete process  page'''
    return render_template('complete.html')

@app.route("/upload",methods=["POST","GET"])
def upload():
    global image_path_screenshot
    if request.method == 'POST':
        img1= request.files.get('file')
        print(img1)
        image_path_screenshot="images/upscaleImages/"+img1.filename
        print('Screen', image_path_screenshot)
        img1.save(image_path_screenshot)
    return render_template('index.html')


@app.route("/",methods=["POST"])
def compare():   
    image = Image.open(image_path_screenshot).convert('RGB')
    print(image)
    start_time = time.time()
    sr_image = model.predict(image)
    sr_image.save('results/{}.jpg')
    
    return render_template('index.html')

# @app.route('/progress')
# def progress():
# 	def generate():
# 		x = 0
# 		while x <= 100:
# 			yield "data:" + str(x) + "\n\n"
# 			x = x + 4
# 			time.sleep(0.5)

# 	#return redirect(url_for('complete')
# 	return Response(generate(), mimetype= 'text/event-stream')

if __name__ == "__main__":
	app.run()