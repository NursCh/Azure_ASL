import requests
import time
from flask import Flask, render_template, Response, jsonify
from PIL import Image
import numpy as np
import io
import threading

app = Flask(__name__)

# Azure Custom Vision API endpoint
API_URL = "https://signlanguagedetector123-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/d46612a9-d79f-40b9-9990-3398d3266afa/classify/iterations/ASL%20GENERAL%20V0.1/image"

# Your Azure prediction key (replace with your actual key)
API_KEY = "Frus0EhlYR4OFxxDhH0A7YjRc8codSwtAx4fbWVUMQOQL0nvth1XJQQJ99AJACYeBjFXJ3w3AAAIACOG5xwX"

detected_label = ""  # Global variable to store the detected label

def process_image():
    global detected_label
    while True:
        # Use the pre-saved image from the static folder
        imageFile = "static/captured.jpg"
        image = Image.open(imageFile)
        image = update_orientation(image)
        image = convert_to_opencv(image)
        h, w = image.shape[:2]
        min_dim = min(w, h)
        max_square_image = crop_center(image, min_dim, min_dim)

        network_input_size = 300
        augmented_image = crop_center(max_square_image, network_input_size, network_input_size)

        # Convert image to bytes for API request
        image_bytes = image_to_bytes(augmented_image)

        # Make the HTTP request to the Azure Custom Vision API
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Prediction-Key": API_KEY
                },
                data=image_bytes
            )
            response.raise_for_status()  # Raise an error for bad responses
            result = response.json()

            # Extract the predicted label from the response
            if result['predictions']:
                detected_label = result['predictions'][0]['tagName']  # The tagName contains the predicted label
                print(detected_label)
            else:
                detected_label = "No prediction"

        except requests.exceptions.RequestException as e:
            print(f"Error during prediction request: {e}")
            detected_label = "Error"
        
        # Sleep for 1 second before processing again
        time.sleep(1)

def image_to_bytes(image):
    """Convert the image to bytes for sending in the API request."""
    img_pil = Image.fromarray(image)  # Convert the numpy array back to PIL Image
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def convert_to_opencv(image):
    """Convert PIL image to OpenCV (BGR) format."""
    image = image.convert('RGB')
    r, g, b = np.array(image).T
    opencv_image = np.array([b, g, r]).transpose()
    return opencv_image

def crop_center(img, cropx, cropy):
    """Crop the image to the center."""
    h, w = img.shape[:2]
    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

def update_orientation(image):
    """Update image orientation based on EXIF data."""
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if exif is not None and exif_orientation_tag in exif:
            orientation = exif.get(exif_orientation_tag, 1)
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

@app.route('/')
def index():
    """Render the homepage with the captured image and detected label."""
    return render_template('index.html', label=detected_label)

@app.route('/video_feed')
def video_feed():
    """Serve the latest captured image."""
    return Response(open('static/captured.jpg', 'rb').read(), mimetype='image/jpeg')

@app.route('/latest_label')
def latest_label():
    global detected_label
    """Return the latest detected label as JSON."""
    label = detected_label
    return jsonify({'label': label})

if __name__ == '__main__':
    # Start the image processing in the background
    threading.Thread(target=process_image, daemon=True).start()

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

