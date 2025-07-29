

# WheatHeadDetectionYOLOv5

Welcome to the Wheat Head Detection project using YOLOv5.

## Objectives
This notebook is focused on detecting wheat heads in field images. I use YOLOv5 (a fast object detector) and techniques like pseudo-labeling and OOF evaluation to improve accuracy.

## Technologies Used
- Python
- PyTorch & YOLOv5 (object detection)
- Pandas, NumPy (data wrangling)
- Matplotlib, OpenCV (visualization)
- Ensemble Boxes (for better bounding boxes)

## Requirements
- Python 3.7+
- torch
- yolov5
- pandas
- numpy
- matplotlib
- opencv-python
- ensemble-boxes



## Input
- JPG images (field photos)
- CSV files with bounding box annotations

## Output
- Trained YOLOv5 model weights
- Detection results (CSV)
- Plots and annotated images

## How to Use
1. Put your images and CSVs in the right folders.
2. Run the notebook to train and predict.
3. Check out the detection results and visualizations.
4. Export your predictions for submission.

---

## Keywords & Techniques Used



**Techniques:**
- Pseudo-labeling
- OOF evaluation
- Bayesian optimization
- Bounding box conversion
- TTA (rotations/flips)
- WBF for ensemble predictions
- Non-max suppression
- Precision/IoU calculation
- Validation
- Submission formatting
- Visualization of predictions vs ground truth
- TTA (test-time augmentation)
