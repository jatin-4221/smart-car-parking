Smart Car Parking

A Smart Car Parking System implemented using Python with machine learning and a Flask web interface.
This project detects vehicles and provides backend support for a smart parking solution.

# Project Overview

Smart Car Parking aims to automate vehicle detection and provide insights for parking management — useful in real-world applications like monitoring space occupancy, guiding drivers, and optimizing parking usage.

The repository includes:

A Flask backend (flaskapp.py) to serve APIs or web interface

Machine learning models (e.g., mAlexnet.py, mALN.py)

Data files such as bounding_boxes_car.json

Scripts possibly related to object detection (maskrcnn.py, maskrcnn_car.py)

Installation

Make sure you have Python 3.8+ installed.

# Install dependencies
pip install -r requirements.txt 

Clone the repository

Create & activate virtual environment (optional)
python3 -m venv venv

source venv/bin/activate   # Linux/Mac

venv\Scripts\activate      # Windows

Install dependencies
pip install -r requirements.txt

pip install flask numpy pandas torch torchvision opencv-python

python flaskapp.py

python mAlexnet.py --train --data data/

python maskrcnn_car.py --image input.jpg

