# yolov5-on-lidar
This repo contains three closely related projects.
1. Fine-tune YoloV5 models for vehicle classification in 2D LiDAR scans by running different experiments with wandb. This is organized inside ./yolov5

![2D bounding box](https://github.com/jselvaraaj/yolov5-on-lidar/blob/main/yolov5_lidar.png?raw=true)

2. Data cleaning and processing labeled LiDAR images for training. This is organized inside ./yolov5 job processing
3. Project and visualized 2D bounding rectangle from yolov5 into 3D bounding cude with [L-Shape fiting](https://github.com/jselvaraaj/3d-registartion-and-l-shape-fitting) algorithm. This is organized inside ./yolov5-prediction-viz.

![3D bounding cube visualization](https://github.com/jselvaraaj/yolov5-on-lidar/blob/main/3d_box_viz.gif?raw=true)

This repo does not contain any of the trained weights, labeled and generated data.