from flask import Flask, render_template, send_file
import os
import cv2
import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import img_to_array


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

class ParkingConfig(Config):
    NAME = "parking_lot"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80 

config = ParkingConfig()
mask_rcnn = MaskRCNN(mode="inference", model_dir="./", config=config)
mask_rcnn.load_weights("mask_rcnn_coco.h5", by_name=True)

resnet_model = ResNet50(weights="imagenet")

bounding_boxes = [] 
results_cache = {} 


def update_bounding_boxes(image_path):
    global bounding_boxes
    image = cv2.imread(image_path)
    results = mask_rcnn.detect([image], verbose=0)[0]

    
    car_boxes = []
    for i, class_id in enumerate(results['class_ids']):
        if class_id == 3:  
            box = results['rois'][i]
            car_boxes.append(box)

    bounding_boxes = car_boxes  


def check_occupancy(image_path):
    global bounding_boxes, results_cache

    image = cv2.imread(image_path)
    occupancy_results = {"occupied": 0, "unoccupied": 0}

    for box in bounding_boxes:
        y1, x1, y2, x2 = box
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (224, 224)) 
        array = img_to_array(resized)
        array = np.expand_dims(array, axis=0)
        array = preprocess_input(array)

        predictions = resnet_model.predict(array, verbose=0)
        label = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=1)[0][0][1]

        if "car" in label:  
            occupancy_results["occupied"] += 1
        else:
            occupancy_results["unoccupied"] += 1

    results_cache = occupancy_results 


@app.route("/")
def home():
    global bounding_boxes, results_cache
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg")

    
    if not os.path.exists(image_path):
        return "Waiting for the first image to be uploaded..."

    
    update_bounding_boxes(image_path)
    check_occupancy(image_path)

  
    image = cv2.imread(image_path)
    for box in bounding_boxes:
        y1, x1, y2, x2 = box
        color = (0, 255, 0)  
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output_image.jpg")
    cv2.imwrite(output_path, image)

    
    total_boxes = len(bounding_boxes)
    occupied = results_cache.get("occupied", 0)
    unoccupied = results_cache.get("unoccupied", 0)

    return render_template(
        "index.html",
        total_boxes=total_boxes,
        occupied=occupied,
        unoccupied=unoccupied,
        image_path="static/output_image.jpg"
    )


@app.route("/image")
def get_image():
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output_image.jpg")
    if os.path.exists(output_path):
        return send_file(output_path, mimetype="image/jpeg")
    else:
        return "Output image not available."


if __name__ == "__main__":
    app.run(debug=True)
