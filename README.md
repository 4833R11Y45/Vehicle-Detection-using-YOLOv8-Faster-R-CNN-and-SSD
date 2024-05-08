# Vehicle-Detection-using-YOLOv8-Faster-R-CNN-and-SSD

![banner](https://github.com/4833R11Y45/Vehicle-Detection-using-YOLOv8-Faster-R-CNN-and-SSD/assets/92614228/10714320-60db-4cf1-9251-1116813a9220)

## Introduction
Object detection is a critical component in the field of computer vision, with applications spanning across traffic monitoring, autonomous driving, and security systems. The advent of deep learning has catalyzed significant advancements in this domain, enhancing both the accuracy and efficiency of detection methods.

This repository is dedicated to the comparison of three state-of-the-art object detection algorithms:

- Faster R-CNN (Region-based Convolutional Neural Network)
- SSD (Single Shot Detector)
- YOLO (You Only Look Once)

Our goal is to evaluate the accuracy, processing time and efficiency of these models to understand their strengths, limitations, and to benchmark their performance metrics.

## Models Overview

### YOLO (You Only Look Once)
YOLO's real-time object detection capabilities have set it apart as a leading algorithm in the field. It utilizes a single-stage detection approach, which means it predicts bounding boxes and class probabilities simultaneously. With its unique architecture, YOLO divides the input image into a grid and assigns bounding boxes and class probabilities for each grid cell.

#### Features:
- Convolutional network architecture that predicts detections in a single pass.
- Continuous improvements with versions YOLOv3, v4, v5, and the latest YOLOv8.
- Enhancements in accuracy through architectural adjustments, anchor boxes, and advanced training techniques.

![image](https://github.com/4833R11Y45/Vehicle-Detection-using-YOLOv8-Faster-R-CNN-and-SSD/assets/92614228/4938ce37-fd8e-4d39-a89c-745c73791fae)

### SSD (Single Shot MultiBox Detector)
SSD stands out for its real-time processing and balanced accuracy, making it a strong competitor against earlier versions of YOLO. It's particularly adept at handling objects of various sizes thanks to its multi-scale feature map approach.

#### Features:
- Single-stage detection model that combines feature maps from different layers.
- Capable of detecting multiple object sizes efficiently.

![image](https://github.com/4833R11Y45/Vehicle-Detection-using-YOLOv8-Faster-R-CNN-and-SSD/assets/92614228/92fe8789-a65d-4450-8fc5-6bcaba5e7778)

### Faster R-CNN (Region-based Convolutional Neural Network)
Faster R-CNN takes a two-stage approach to object detection, prioritizing accuracy with a more complex computational process. It first generates region proposals through an RPN and then uses a Fast R-CNN network for precise detection.

#### Features:
- Two-stage detection process with a Region Proposal Network (RPN).
- Utilizes backbone networks like ResNet or MobileNet for feature extraction.
- Known for its high precision in object detection tasks.

![image](https://github.com/4833R11Y45/Vehicle-Detection-using-YOLOv8-Faster-R-CNN-and-SSD/assets/92614228/0cd96ea4-00bd-48c6-a0fc-7d90c9e62012)

## Implementation Details

All three models were implemented using the PyTorch framework. We utilized pretrained weights to enhance the initial accuracy and training efficiency:

- YOLOv8s model with CSPDarkNet backbone.
- Faster R-CNN model with MobileNet v3 Large 320 FPN backbone.
- SSD model with MobileNet v3 Large backbone.

The models were trained using the following hyperparameters:

- Epochs: 300
- Learning rate: 0.01
- Batch size: 16

This configuration ensured that each model was trained under the same conditions, allowing for a fair comparison of their performance.

## Repository Contents

- `Detection Results/` - Object Detection visualizations by the models.
- `Models/` - Weights of the trained models.
- `Notebooks/` - Implementation of the models.

## Getting Started

To get started with the models, follow the instructions in each notebook.

## Datasets

The datasets used for this study can be found at the following links:

- [Vehicle Detection Dataset](https://universe.roboflow.com/cvproject-y6bf4/vehicle-detection-gr77r) - A dataset comprising various vehicle types, utilized for training object detection models.
- [VISDRONE Dataset](https://universe.roboflow.com/dataset-conversion-ipkwb/visdrone-uhzsx) - A comprehensive dataset for visual drone detection, providing a diverse set of aerial images.
- [Pak Vehicles Dataset](https://universe.roboflow.com/pak-vehicles/pak-vehicles/model/2) - A self-prepared dataset depicting the congested vehicle traffic situation in Pakistan from UAV view.

### Evaluation Results
The models were trained and evaluated on three different datasets, providing insights into their performance across various metrics such as inference time, mean Average Precision (mAP) at different IoU thresholds, and Average Recall (AR).

#### YOLOv8s
- **Trained on the first dataset:**
  - Recall: 0.896
  - mAP50: 0.946
  - mAP50-95: 0.876
- **Trained on the second dataset:**
  - Recall: 0.669
  - mAP50: 0.743
  - mAP50-95: 0.466
- **Trained on the third dataset:**
  - Recall: 0.699
  - mAP50: 0.766
  - mAP50-95:  0.336

![YOLOv8s](https://github.com/4833R11Y45/Vehicle-Detection-using-YOLOv8-Faster-R-CNN-and-SSD/blob/main/Detection%20Results/YOLOv8s.jpg)

#### Faster R-CNN
- **Trained on the first dataset:**
  - AP @ IoU=0.50:0.95: 0.754
  - AP @ IoU=0.50: 0.895
  - AR @ IoU=0.50:0.95: 0.800
- **Trained on the second dataset:**
  - AP @ IoU=0.50:0.95: 0.053
  - AP @ IoU=0.50: 0.131
  - AR @ IoU=0.50:0.95: 0.090
- **Trained on the third dataset:**
  - AP @ IoU=0.50:0.95: 0.073
  - AP @ IoU=0.50: 0.23
  - AR @ IoU=0.50:0.95: 0.124

![Faster C-NN](https://github.com/4833R11Y45/Vehicle-Detection-using-YOLOv8-Faster-R-CNN-and-SSD/blob/main/Detection%20Results/Faster%20R-CNN.png)

#### SSD
- **Trained on the first dataset:**
  - AP @ IoU=0.50:0.95: 0.531
  - AP @ IoU=0.50: 0.824
  - AR @ IoU=0.50:0.95: 0.674
- **Trained on the second dataset:**
  - AP @ IoU=0.50:0.95: 0.025
  - AP @ IoU=0.50: 0.067
  - AR @ IoU=0.50:0.95: 0.120
- **Trained on the third dataset:**
  - AP @ IoU=0.50:0.95: 0.042
  - AP @ IoU=0.50: 0.152
  - AR @ IoU=0.50:0.95: 0.135
 
![SSD](https://github.com/4833R11Y45/Vehicle-Detection-using-YOLOv8-Faster-R-CNN-and-SSD/blob/main/Detection%20Results/SSD.png)

These evaluations showcase the comparative strengths of each model and highlight their capabilities in different scenarios.

## Contributing

We welcome contributions to improve the models and extend the comparative analysis. If you have suggestions or improvements, please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Citations

If you utilize the datasets provided, please cite the following:

```bibtex
@misc{ vehicle-detection-gr77r_dataset,
    title = { Vehicle Detection Dataset },
    type = { Open Source Dataset },
    author = { CVproject },
    howpublished = { \url{ https://universe.roboflow.com/cvproject-y6bf4/vehicle-detection-gr77r } },
    url = { https://universe.roboflow.com/cvproject-y6bf4/vehicle-detection-gr77r },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2022 },
    month = { dec },
    note = { visited on 2024-02-07 },
}
```

```bibtex
@misc{ visdrone-uhzsx_dataset,
    title = { VISDRONE Dataset },
    type = { Open Source Dataset },
    author = { Dataset Conversion },
    howpublished = { \url{ https://universe.roboflow.com/dataset-conversion-ipkwb/visdrone-uhzsx } },
    url = { https://universe.roboflow.com/dataset-conversion-ipkwb/visdrone-uhzsx },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2022 },
    month = { aug },
    note = { visited on 2024-02-07 },
}
```
```bibtex
@misc{ pak-vehicles_dataset,
    title = { Pak Vehicles Dataset },
    type = { Open Source Dataset },
    author = { Pak Vehicles },
    howpublished = { \url{ https://universe.roboflow.com/pak-vehicles/pak-vehicles } },
    url = { https://universe.roboflow.com/pak-vehicles/pak-vehicles },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2024 },
    month = { may },
    note = { visited on 2024-05-08 },
}
```
