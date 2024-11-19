import cv2
import base64
import time
from azure.iot.hub import IoTHubRegistryManager

# Replace with your IoT Hub connection string and device ID
CONNECTION_STRING = "HostName=ASLDeviceManager.azure-devices.net;SharedAccessKeyName=iothubowner;SharedAccessKey=ThkDfYTcZn0zX1/9XY8j19pEd5/LRRoTUAIoTNNniFg="
DEVICE_ID = "Receiver"  # The device you want to send the C2D message to

# Create the registry manager to send C2D messages
registry_manager = IoTHubRegistryManager(CONNECTION_STRING)

def capture_image():
    """Capture an image from the webcam and return it as a base64 string."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap.release()
        return None
    
    frame_resized = cv2.resize(frame, (320, 240))

    _, encoded_image = cv2.imencode('.jpg',  frame_resized)
    base64_image = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

    cv2.imshow("Image",  frame_resized)
    cv2.waitKey(1)

    cap.release()
    return base64_image

def send_message_to_device(image_data):
    """Send a C2D message to the specified device."""
    try:
        registry_manager.send_c2d_message(DEVICE_ID, image_data)  # Send C2D message to receiver
        print(f"C2D message sent to device {DEVICE_ID}")
    except Exception as e:
        print(f"Error sending C2D message: {e}")

try:
    print("Sending images to IoT Hub...")
    while True:
        image = capture_image()
        if image:
            send_message_to_device(image)  # Send the captured image as a C2D message
        time.sleep(2)  # Send every 2 seconds
except KeyboardInterrupt:
    print("Stopped by user.")

