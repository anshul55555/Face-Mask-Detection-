# Face Mask Detection using CNN

## Project Overview
This project builds a **Face Mask Detection** system using a **Convolutional Neural Network (CNN)** to classify images into two categories:
- **With Mask**
- **Without Mask**

The main goal is to create a simple and effective deep learning model that can automatically identify whether a person is wearing a face mask from an input image. This type of system can be useful in public safety monitoring, hospitals, airports, malls, and surveillance-based entry systems.

---

## Problem Statement
Face mask detection is a **binary image classification** problem. The model takes a face image as input and predicts whether the person is wearing a mask or not. The project follows a complete deep learning workflow starting from dataset collection and preprocessing to model training, evaluation, and prediction.

---

## Dataset
The dataset is taken from Kaggle:
**Face Mask Dataset** by `omkargurav`

It is downloaded inside the notebook using:

```python
import kagglehub
path = kagglehub.dataset_download("omkargurav/face-mask-dataset")
```

The dataset contains two image classes:
- `with_mask`
- `without_mask`

---

## Tools and Libraries Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow / Keras
- Scikit-learn
- KaggleHub

---

## Project Workflow
### 1. Data Collection
The dataset is downloaded directly from Kaggle and the folder structure is checked to identify the class-wise image directories.

### 2. Data Preparation
- Image paths are collected into a dataframe
- Labels are assigned based on folder names
- Class distribution is checked
- Sample images are displayed for visual understanding

### 3. Train, Validation, and Test Split
The dataset is split into:
- **70% Training**
- **15% Validation**
- **15% Testing**

This helps train the model properly and evaluate it fairly.

### 4. Image Preprocessing and Augmentation
The images are:
- resized to a fixed shape
- normalized automatically
- augmented using transformations like rotation, zoom, and horizontal flip

This improves the model’s ability to generalize on unseen images.

### 5. CNN Model Building
A custom CNN model is created using:
- Convolution layers
- ReLU activation
- MaxPooling layers
- Dropout layers
- Dense layers
- Sigmoid output layer

The model is compiled using:
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metric:** Accuracy

### 6. Model Training
The CNN is trained on the training dataset and validated on the validation set. Callbacks are used to improve training and save the best model performance.

### 7. Model Evaluation
The final model is tested using:
- Test Accuracy
- Confusion Matrix
- Classification Report
- Precision and Recall

### 8. Single Image Prediction
A helper function is included in the notebook to test the model on one image and display the predicted class.

### 9. Model Saving
The trained model is saved in `.keras` format so it can be reused later without retraining.

---

## Output
The project produces:
- a trained CNN model for mask detection
- performance graphs for training and validation
- confusion matrix and classification metrics
- sample predictions on test images
- saved final model file

---

## Notebook Sections
The notebook is organized into these sections:
1. Project objective
2. Download the dataset from Kaggle
3. Build a dataframe of image paths
4. Show a few sample images
5. Train / validation / test split
6. Image preprocessing and augmentation
7. Build the CNN model
8. Callbacks
9. Train the model
10. Plot training history
11. Evaluate on the test set
12. Try prediction on a single image
13. Save the trained model
14. Final conclusion

---

## Conclusion
This project demonstrates how CNN can be used for a real-world image classification task like face mask detection. It covers the full pipeline from raw image data to model evaluation and prediction. The project is simple, practical, and useful for understanding how deep learning can be applied to safety-related computer vision problems.

---
