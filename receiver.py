import base64
import os
import time
import cv2
import numpy as np
from azure.iot.device import IoTHubDeviceClient

# Replace with your Azure IoT Hub device connection string
CONNECTION_STRING = "HostName=ASLDeviceManager.azure-devices.net;DeviceId=Receiver;SharedAccessKey=RWtlJlDiNkiqcwRzzjyTArthyK6bopgrY7AzOY/kEOM="

# Folder to save uploaded images
UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Create the IoT Hub client
device_client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)

def decode_and_save_image(base64_image):
    """Decode base64 image and save it to the static folder."""
    try:
        # Decode the base64 string into bytes
        img_data = base64.b64decode(base64_image)

        # Convert bytes to numpy array and decode image
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save the image in the static/images folder
        image_path = os.path.join(UPLOAD_FOLDER, "captured.jpg")
        cv2.imwrite(image_path, img)

        print(f"Image saved at {image_path}")
    except Exception as e:
        print(f"Error decoding and saving image: {e}")

def on_message_received(message):
    """Callback to process received message from IoT Hub."""
    try:
        # Decode the received message data
        message_data = message.data.decode("utf-8")

        # Save the image by decoding the base64 image data
        decode_and_save_image(message_data)

    except Exception as e:
        print(f"Error processing the message: {e}")

# Set the callback function for message handling
device_client.on_message_received = on_message_received

def listen_for_messages():
    """Listen for incoming messages."""
    print("Listening for messages from IoT Hub...")
    try:
        while True:
            time.sleep(1)  # Keep the receiver running to listen for messages

    except KeyboardInterrupt:
        print("Receiver stopped by user.")
    finally:
        device_client.disconnect()  # Disconnect when done

# Connect to IoT Hub and start listening for messages
device_client.connect()
listen_for_messages()

