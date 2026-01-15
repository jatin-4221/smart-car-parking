import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.utils import extract_bboxes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class InferenceConfig(Config):
    NAME = "car_detection"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir="./", config=config)

coco_weights_path = "mask_rcnn_coco.h5"
model.load_weights(coco_weights_path, by_name=True)

image_path = "./test4.jpg" 
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model.detect([image_rgb], verbose=1)
r = results[0]

bounding_boxes = extract_bboxes(r['masks'])

min_confidence = 0.5

output_path = "bounding_boxes_car.json"
boxes_list = []
confidence_scores = []

for i, box in enumerate(bounding_boxes):
    if r['class_ids'][i] == 3 and r['scores'][i] >= min_confidence: 
        boxes_list.append({
            "id": i,
            "bounding_box": box.tolist(),
            "class_id": int(r['class_ids'][i]),
            "score": float(r['scores'][i])
        })
        confidence_scores.append(r['scores'][i])  


if confidence_scores:
    avg_confidence = np.mean(confidence_scores)
else:
    avg_confidence = 0.0

with open(output_path, "w") as f:
    json.dump(boxes_list, f, indent=4)


for i, box in enumerate(bounding_boxes):
    if r['class_ids'][i] == 3 and r['scores'][i] >= min_confidence:
        y1, x1, y2, x2 = box
        color = (0, 255, 0)  
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"Car: {r['scores'][i]:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

output_image_path = "output_with_boxes_car.jpg"
cv2.imwrite(output_image_path, image)

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Cars with Bounding Boxes")
plt.show()

print(f"Bounding boxes for cars saved to {output_path}")
print(f"Annotated image with cars saved to {output_image_path}")
print(f"Average confidence for detected cars: {avg_confidence:.2f}")
