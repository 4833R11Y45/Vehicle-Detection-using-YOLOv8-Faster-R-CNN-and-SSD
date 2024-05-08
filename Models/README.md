# Models Directory

This directory contains the trained models for the object detection project, "Vehicle Detection using YOLOv8, Faster R-CNN, and SSD" Each model was trained using two distinct datasets to compare performance across different object detection frameworks: YOLOv8s, Faster R-CNN, and SSD.

## YOLOv8s Models

- `yolov8s_best_v1.pt`: Weights of the YOLOv8s model trained on the first dataset. This model showcases high precision and real-time detection capabilities optimized for vehicle detection.
- `yolov8s_best_v2.pt`: Weights of the YOLOv8s model trained on the second dataset. Adapted for more extensive and varied data, showcasing the scalability of the model.
- `yolov8s_best_v2.pt`: Weights of the YOLOv8s model trained on the third dataset. Adapted for Pakistani traffic situation, also showing capabilities of detecting bikes and Rickshaws.

## Faster R-CNN Models

The weights for the Faster R-CNN models are hosted on OneDrive due to their size. You can download them using the following link:

[OneDrive - Faster R-CNN Models](https://1drv.ms/f/s!At94amvVQIfig_V_JTbR-_zsFhuaLg?e=XppbQD)

- `faster_rcnn_vehicle.pth`: Weights of the Faster R-CNN model trained on the first dataset. Known for its high accuracy due to the two-stage detection process.
- `faster_rcnn_vehicle_ultimatum.pth`: Weights of the Faster R-CNN model trained on the second dataset. Provides insight into the model's performance on a more challenging and diverse dataset.
- `faster_rcnn_vehicle_final_updated.pt`: Weights of the Faster R-CNN model trained on the third dataset. Adapted for Pakistani traffic situation, also showing capabilities of detecting bikes and Rickshaws.

## SSD Models

- `ssd_vehicle_v1.pth`: Weights of the SSD model trained on the first dataset. This model offers a trade-off balance between speed and accuracy, suitable for real-time applications with less computational resources.
- `ssd_vehicle_v2.pth`: Weights of the SSD model trained on the second dataset. Demonstrates the model's adaptability and performance across different scales of object sizes.
- `ssd_vehicle_full_final.pt`: Weights of the SSD model trained on the third dataset. Adapted for Pakistani traffic situation, also showing capabilities of detecting bikes and Rickshaws.

## Usage

To use these models for inference or further training, download them and integrate them in model implementation notebooks provided in the `Notebooks/` dorectory.

## Contributing

If you have made improvements or have suggestions regarding these models, please feel free to submit a pull request or open an issue to discuss your changes.

Thank you for exploring our models for object detection. We hope these resources will aid in your projects and research endeavors.
