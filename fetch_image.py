import time
import cv2

VIDEO_URL = "http://192.168.1.2:8080/video" 
OUTPUT_IMAGE = "static/image.jpg"  

def fetch_image():
    """Fetch an image from the video feed every 30 seconds."""
    while True:
        try:
            cap = cv2.VideoCapture(VIDEO_URL)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(OUTPUT_IMAGE, frame)
                print(f"Saved image to {OUTPUT_IMAGE}")
            else:
                print("Failed to fetch frame from video feed.")
            cap.release()
        except Exception as e:
            print(f"Error fetching image: {e}")
        time.sleep(30) 

if __name__ == "__main__":
    fetch_image()
