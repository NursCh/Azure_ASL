from flask import Flask, render_template, Response, jsonify
import cv2
from PIL import Image
import time
import tensorflow as tf
import numpy as np

app = Flask(__name__)

output_layer = 'model_output:0'
input_node = 'data:0'

graph_def = tf.compat.v1.GraphDef()
labels = []

# These are set to the default names from exported models, update as needed.
filename = "model/model.pb"
labels_filename = "model/labels.txt"

with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

detected_label = ""  # Global variable to store the detected label

# Create a TensorFlow session
sess = tf.compat.v1.Session()
prob_tensor = sess.graph.get_tensor_by_name(output_layer)

def process_image():
    global detected_label
    while True:
        imageFile = "static/captured.jpg"  # Use the pre-saved image
        try:
            image = Image.open(imageFile)
        except FileNotFoundError:
            print(f"Image file {imageFile} not found.")
            time.sleep(1)
            continue

        image = update_orientation(image)
        image = convert_to_opencv(image)
        image = resize_down_to_1600_max_dim(image)
        h, w = image.shape[:2]
        min_dim = min(w, h)
        max_square_image = crop_center(image, min_dim, min_dim)
        augmented_image = resize_to_256_square(max_square_image)

        network_input_size = 300

        augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

        try:
            predictions = sess.run(prob_tensor, {input_node: [augmented_image] })
        except KeyError:
            print("Couldn't find classification output layer: " + output_layer + ".")
            print("Verify this is a model exported from an Object Detection project.")
            exit(-1)

        highest_probability_index = np.argmax(predictions)
        detected_label = labels[highest_probability_index]

        time.sleep(1)  # Wait for 1 second before processing the image again

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r, g, b = np.array(image).T
    opencv_image = np.array([b, g, r]).transpose()
    return opencv_image

def crop_center(img, cropx, cropy):
    h, w = img.shape[:2]
    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)

def update_orientation(image):
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
    import threading
    threading.Thread(target=process_image, daemon=True).start()

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

