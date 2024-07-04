from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io
from skimage.transform import resize
import numpy as np
import cv2
import base64
import pickle

tf.config.set_visible_devices([], 'CPU')

app = Flask(__name__)

model=pickle.load(open('models/model.pkl','rb'))

# Load the saved model
model = tf.keras.models.load_model('model_for_nuclei.keras', custom_objects={'Lambda': tf.keras.layers.Lambda}, compile=False, safe_mode=False)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})


    
    image_file = request.files['image']
    image = io.imread(image_file)
    test_data = np.expand_dims(image, axis=0)
    

    # Make predictions on the test data
    pred = model.predict(test_data)
    pred = pred[0]

    # Convert predicted mask to binary
    threshold = 0.16  # Define the threshold value
    binary_mask = np.where(pred > threshold, 1, 0)

    # Convert the binary mask to uint8 for visualization
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)

    # Apply color to the binary mask
    mask_colored = cv2.cvtColor(binary_mask_uint8, cv2.COLOR_GRAY2BGR)
    mask_color = (0, 0, 255)  # Red color in BGR format

    masked_areas = np.where(mask_colored == 255)  # Find the masked areas
    mask_colored[masked_areas[0], masked_areas[1], :] = mask_color 

    masked_image = cv2.addWeighted(image.astype(np.float32), 0.5, mask_colored.astype(np.float32), 0.5, 0)
    masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)
    

    # Convert the masked image to base64 for embedding in HTML
    _, img_buffer = cv2.imencode('.png', masked_image)
    img_base64 = base64.b64encode(img_buffer).decode('utf-8')

    return jsonify({'result_image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
    
    

