import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.utils import extract_bboxes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class InferenceConfig(Config):
    NAME = "car_detection"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO has 80 classes + background

config = InferenceConfig()
config.display()

# Initialize the Mask R-CNN model
model = modellib.MaskRCNN(mode="inference", model_dir="./", config=config)

# Load the pre-trained COCO weights
coco_weights_path = "mask_rcnn_coco.h5"
model.load_weights(coco_weights_path, by_name=True)

# Load and preprocess the image
image_path = ".\static\image.jpg"  # Replace with the path to your image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run inference
results = model.detect([image_rgb], verbose=1)
r = results[0]

# Extract bounding boxes
bounding_boxes = extract_bboxes(r['masks'])

# Save bounding boxes to a file
output_path = "bounding_boxes.json"
boxes_list = []
for i, box in enumerate(bounding_boxes):
    boxes_list.append({
        "id": i,
        "bounding_box": box.tolist(),
        "class_id": int(r['class_ids'][i]),
        "score": float(r['scores'][i])
    })

with open(output_path, "w") as f:
    json.dump(boxes_list, f, indent=4)

# Draw bounding boxes and save the visualization
for i, box in enumerate(bounding_boxes):
    y1, x1, y2, x2 = box
    color = (0, 255, 0)  # Green bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    class_id = r['class_ids'][i]
    label = f"{class_id}: {r['scores'][i]:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

output_image_path = "output_with_boxes.jpg"
cv2.imwrite(output_image_path, image)

# Display the resulting image
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Objects with Bounding Boxes")
plt.show()

print(f"Bounding boxes saved to {output_path}")
print(f"Annotated image saved to {output_image_path}")
