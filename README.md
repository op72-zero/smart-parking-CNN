# CNN Parking Occupancy Detection

This project uses a Convolutional Neural Network (CNN) to detect parking spot occupancy from images. The model classifies images as either **Occupied** or **Empty**.

---

## 📁 Folder Structure

parking-cnn-project/
│
├── raw_dataset.zip # Compressed raw dataset (contains 'Occupied/' and 'Empty/' images)
├── models/
│ └── parking_cnn.h5 # Trained CNN model
├── src/
│ ├── preprocess.py # Data preprocessing script
│ ├── train_model.py # Model training script
│ └── test_model.py # Testing / inference script
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## Required Packages

Install all necessary Python packages using pip:

pip install tensorflow numpy pandas opencv-python matplotlib scikit-learn
Setup Instructions
Download & unzip dataset

Extract raw_dataset.zip into the project root folder:
raw_dataset/
    Occupied/
    Empty/
    
Preprocess the data

Run the preprocessing script to prepare images for training:
python src/preprocess.py

Train the CNN model (optional if you already have parking_cnn.h5)
python src/train_model.py
Test / Inference

Run the test script to predict occupancy on new images:
python src/predict.py
📦 Dataset
The dataset contains images of parking spots categorized as Occupied and Empty.

For convenience, the dataset is compressed as raw_dataset.zip.

Only the raw dataset is included — preprocessing will be done by preprocess.py.

Notes:
Ensure you have TensorFlow installed (version 2.x recommended).

If the model file parking_cnn.h5 is not present, run training first.

Keep the folder structure intact for scripts to work properly.
