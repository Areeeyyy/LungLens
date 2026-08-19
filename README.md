# LungLens

**Pneumonia Classification on Chest X-Ray Images Using Convolutional Neural Network (CNN) Based Web Application**

LungLens is an interactive web application that leverages a Convolutional Neural Network (CNN) algorithm to classify chest X-Ray images into two categories: **NORMAL** and **PNEUMONIA**. The application is equipped with an Explainable AI feature based on Grad-CAM (Gradient-weighted Class Activation Mapping) that generates heatmap visualizations highlighting the model's focus areas, thereby enhancing prediction transparency and interpretability.

> **Disclaimer:** This application is developed for educational and research purposes only. Prediction results are **not** a substitute for professional medical diagnosis.

## Key Features

| No | Feature | Description |
|:--:|---------|-------------|
| 1 | Automated Classification | Classifies chest X-Ray images into NORMAL or PNEUMONIA |
| 2 | Prediction Probability | Displays probability values (0 to 1) and confidence percentage |
| 3 | Grad-CAM Visualization | Renders heatmap of the model's focus areas on the X-Ray image |
| 4 | Multi-Page Interface | Consists of Home, Upload, and How It Works pages |
| 5 | Real-Time Inference | Prediction results are displayed instantly after image upload |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| Web Framework | Streamlit |
| Explainable AI | Grad-CAM (tf.GradientTape) |
| Image Processing | Pillow (PIL), NumPy |
| Methodology | CRISP-DM |

## Model Performance

The CNN model was trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle, consisting of 5,856 pediatric patient images.

| Metric | Score |
|--------|:-----:|
| Accuracy | 93% |
| Precision (PNEUMONIA) | 97% |
| Recall (PNEUMONIA) | 93% |
| F1-Score (PNEUMONIA) | 95% |

### CNN Architecture

```
Input (150 x 150 x 3, RGB Image)
    |
    +-- Conv2D(32, 3x3)  -> ReLU -> MaxPool(2x2)
    |
    +-- Conv2D(64, 3x3)  -> ReLU -> MaxPool(2x2)
    |
    +-- Conv2D(128, 3x3) -> ReLU -> MaxPool(2x2)
    |
    +-- Conv2D(128, 3x3) -> ReLU -> MaxPool(2x2)
    |
    +-- Conv2D(256, 3x3) -> ReLU -> MaxPool(2x2)
    |
    +-- Flatten
    |
    +-- Dense(512, ReLU) -> Dropout(0.5)
    |
    +-- Dense(1, Sigmoid) -> Output [0...1]
```

Total trainable parameters: **1,061,313**

## Project Structure

```
app/
├── main.py                    # Streamlit application entry point
├── pneumonia_cnn_model.h5     # Trained CNN model
├── requirements.txt           # Python dependencies
├── pages/
│   ├── home.py                # Home page
│   ├── upload.py              # Prediction page (Upload)
│   ├── how_it_works.py        # How It Works page
│   └── gradcam.py             # Grad-CAM module (Explainable AI)
└── styles/
    └── main.css               # UI stylesheet
```

## Installation and Usage

### Prerequisites

* Python 3.9, 3.10, or 3.11
* pip (Python package manager)

### Installation Steps

1. Clone this repository:

```bash
git clone https://github.com/username/LungLens.git
```

2. Navigate to the project directory:

```bash
cd LungLens/app
```

3. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Ensure the model file `pneumonia_cnn_model.h5` is placed inside the `app/` directory.

### Running the Application

```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`. Open this address in your browser.

## Application Overview

### Home Page

This page serves as the main entry point, presenting a brief description of the system along with navigation buttons to access the prediction feature and the technical explanation page.

### Prediction Page (Upload)

Users upload a chest X-Ray image through the file uploader component. Once processed, results are displayed in a three-column layout:

* **Left Column:** Original uploaded X-Ray image
* **Center Column:** Grad-CAM heatmap visualization (red = high focus, blue = low focus)
* **Right Column:** Classification label, confidence level, and probability bars for both classes

### How It Works Page

An educational page that explains the CNN architecture, the function of each layer, the training pipeline (CRISP-DM), and the prediction workflow.

## Prediction Pipeline

When a user uploads an image, the system executes eight sequential stages:

1. Image upload (accepts PNG, JPG, or JPEG formats)
2. RGB format conversion (3 channels)
3. Resizing to 150x150 pixels
4. Pixel normalization (divided by 255, range 0 to 1)
5. CNN model inference
6. Output probability calculation
7. Gradient tracing and Grad-CAM heatmap generation
8. Classification result display with visualizations

## References

* Kermany, D. S. et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-based Deep Learning." *Cell*, 172(5), 1122-1131.
* Selvaraju, R. R. et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *Proceedings of the IEEE ICCV*, 618-626.
* Mooney, P. (2018). "Chest X-Ray Images (Pneumonia)" [Dataset]. Kaggle.

