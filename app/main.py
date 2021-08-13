# run by typing python3 main.py in a terminal 
import os
import cv2
import torch
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from model import ConvNet, Generator
from predict import predict_ingredients, binarize, soft, reconstruct_images_gan, get_labels, NUM_PICS
from utils import get_base_url, allowed_file, and_syntax

# setup the webserver
'''
    coding center code
    port may need to be changed if there are multiple flask servers running on same server
    comment out below three lines of code when ready for production deployment
'''
# port = 12349
# base_url = get_base_url(port)
app = Flask(__name__)

'''
    cv scaffold code
    uncomment below line when ready for production s
'''
# app = Flask(__name__)

UPLOAD_FOLDER = 'images'
NUM_CLASSES = 353
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

ingredient_pred_model = ConvNet(NUM_CLASSES)
image_recon_model = Generator()
ingredient_pred_model.load_state_dict(torch.load("weights/ingredient_pred.pth", map_location=torch.device('cpu')))
image_recon_model.load_state_dict(torch.load("weights/image_recon.pth", map_location=torch.device('cpu')))
ingredient_pred_model.eval()
image_recon_model.eval()


@app.route('/')
# @app.route(base_url)
def home():
    return render_template('Home.html')

@app.route('/', methods=['POST'])
# @app.route(base_url, methods=['POST'])
def home_post():
    global model

    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        response_dict = {"original_image_path": image_path}
        ingredients = predict_ingredients(image_path, ingredient_pred_model)
        ingredients_binary = soft(ingredients)

        if ingredients_binary.sum() == 0:
            response_dict["ingredients"] = "No ingredients detected"
            return response_dict
        else:
            # get labels from vectors and produce ingredients string
            list_of_labels = get_labels(binarize(ingredients))
            descriptor_ingredients_string = and_syntax(list_of_labels)
            response_dict["ingredients"] = descriptor_ingredients_string

            # get image reconstructions whose filepaths will be sent to frontend
            reconstructed_images = reconstruct_images_gan(ingredients_binary, image_recon_model)
            reconstructed_image_paths = []
            parts_of_filename = filename.split(".")

            for i in range(NUM_PICS):
                new_filename = "".join(parts_of_filename[:-1]) + \
                               f"{i}." + parts_of_filename[-1]
                secure_new_filename = secure_filename(new_filename)
                new_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_new_filename)
                output_image = reconstructed_images[i].permute(1,2,0).numpy()
                cv2.imwrite(new_image_path, cv2.resize(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR), (224, 224)))
                url_for_filename = url_for('files', filename=secure_new_filename)
                reconstructed_image_paths.append(url_for_filename)
            response_dict["reconstructed_image_paths"] = reconstructed_image_paths

            return response_dict
    else:
        return redirect(url_for('home'))

@app.route('/files/<path:filename>')
# @app.route(base_url + '/files/<path:filename>')
def files(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment = True)

if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalc2.ai-camp.org to the site where you are editing this file.
    website_url = 'cocalc4.ai-camp.org'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    print(app.root_path)
    # remove debug=True when deploying it
    app.run(host = '0.0.0.0', port = port, debug=True)
    import sys; sys.exit(0)

    '''
    cv scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)
