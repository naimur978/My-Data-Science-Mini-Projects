

# UNetSemanticSegmentation

Welcome to the U-Net Semantic Segmentation project.

## Objectives
This notebook addresses the salt identification challenge using the U-Net architecture. My goal is to segment images and find salt deposits, using deep learning and some classic image processing.

## Technologies Used
- Python
- Keras & TensorFlow (for building and training U-Net)
- NumPy, Pandas (data handling)
- Matplotlib, Scikit-image (visualization & image ops)

## Requirements
- Python 3.7+
- keras
- tensorflow
- numpy
- pandas
- matplotlib
- scikit-image


## Input
- PNG images (raw data)
- PNG masks (ground truth for segmentation)

## Output
- Trained U-Net model files
- Predicted segmentation masks
- Submission CSVs
- Plots and sample images

## How to Use
1. Place your images and masks in the correct folders.
2. Run the notebook to preprocess, train, and predict.
3. Check out the visualizations and segmentation results.
4. Export your predictions for submission.

---

## Keywords & Techniques Used


**Techniques:**
- Data preprocessing
- Resizing images/masks
- Boolean mask handling
- Sigmoid activation
- MeanIoU metric
- Early stopping
- Model checkpointing
- ReduceLROnPlateau
- Thresholding predictions
- Upsampling masks
- Run-length encoding for submission
- Sanity checks with visualization

