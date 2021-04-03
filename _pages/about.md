---
permalink: /about/
title: "About Me"
layout: single
header:
  overlay_image: assets/utils/tree.gif
  # video:
  #   width: 70%
  #   id: 9dfWzp7rYR4
  #   provider: youtube
  caption: "Photo credit: [**Tenor**](https://tenor.com/search/animated-van-gogh-gifs)"
excerpt: >
  The kingdom of heaven is like treasure hidden in a field. <br /> When a man found it, he hid it again, <br /> and then in his joy went and sold all he had and bought that field.<br />
  - Matthew 13:44
toc: true
toc_sticky: true
comments: true
author_profile: true
---
## Biography

Hello 🎅🏼, my name is Jie Yuan. Welcome to my site.

I am a graduate of the [Leibniz University Hannover](https://www.uni-hannover.de/) program [Navigation and Field Robotics](https://www.uni-hannover.de/en/studium/studienangebot/info/studiengang/detail/navigation-and-field-robotics/) specializing in Simultaneous Localization and Mapping(SLAM) and Computer Vision based on Deep Learning. Python(for deep learning) and C++(for SLAM) are my principal developing languages, and sometimes I use Matlab to verify algorithms.

I develop multi-sensor (camera, radar, IMU, GPS) perception and filter-based localization for mobile robotic. Besides, deep-learning-based scene understanding and reinforcement learning for automatic control is also a primary concern in my study. My university is both a production and a learning process - working in teams to produce practical applications supporting autonomous driving. The relevant areas are listed below.

* __HD Mapping__
* __Robot Localization__
* __Robot Motion Planning__
* __3D Scene Simulation__
* __Object Tracking__
* __Sensor Calibration__
* __Image Annotation__
* __Scene Segmentation__


If you want to know whether a penguin can fly or not, the answer is NO! Following video might be the greatest Fools pranks of all time of BBC.

{% include video id="9dfWzp7rYR4" provider="youtube" %}

Please feel free to contact me [📧](mailto:yuanjielovejesus@gmail.com).

## Skills

I am fond of cooking 👩‍🍳🍚🏺🥠. If it is possible to select a second occupation, it must be a cook. I subscribe youtube channels [Chef Wang](https://www.youtube.com/channel/UCg0m_Ah8P_MQbnn77-vYnYw), [老饭骨](https://www.youtube.com/channel/UCBJmYv3Vf_tKcQr5_qmayXg), and [小高姐的 Magic Ingredients](https://www.youtube.com/channel/UCCKlp1JI9Yg3-cUjKPdD3mw).


## Technique Books

Here are books 📚 that I like reviews a lot during development:

* [Probabilistic Robotics](https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf)
* [State Estimation for Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf)
* [Multiple View Geometry in Computer Vision](https://www.amazon.com/Multiple-View-Geometry-Computer-Vision/dp/0521540518)
* [Deep Learning](https://www.deeplearningbook.org/)
* [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)


## Projects

The following are some projects that are done as student or student research assistants. 

### Digital Earth based on WMS [link](https://github.com/penguinflys/Oriental_EYE)

![image-center]({{ "/assets/images/digitalearth.png" | relative_url }}){: .align-center style="width: 100%;"}
*Fig. 0 Visualization of Digital Earth with DEM model*
{: .text-center}

This project is a teaching project which simulates the elevation of the earth and the depth of the ocean. It is initially based on the [icosahedron](https://en.wikipedia.org/wiki/Icosahedron) and iteratively refined in a deeper level to sample the local elevation. It can also simulate the gravity field if the data is in hand.

Techniques: OpenGL, C++, Geographic Grid.

### LEGO Courier Student Toy Project

{% include video id="Rj_TkF2gSKw?start=60" provider="youtube" %}

This is the first project that requires knowledge of SLAM. It is developed in C++ with ROS as a communication tool, PCL, and OpenCV as data processing API. It is conducted in a closed arena with lots of poles and a short tunnel. The examiner set the arena scenario, delivering destination and courier born place randomly, which requires the robustness of the method. What I developed is in the last of the video, namely "SLAM method." The development can be mainly divided into the following pieces.

<table>
  <thead>
    <tr>
      <th style="text-align: center">Internal Sensors</th>
      <th style="text-align: center">External Sensors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">LIDAR</td>
      <td style="text-align: center" rowspan=3>Camera</td>
    </tr>
    <tr>
      <td style="text-align: center">Ultrasonic Unit</td>
    </tr>
    <tr>
      <td style="text-align: center">Odometry(broken)</td>
    </tr>
  </tbody>
</table>

<!-- | Internal Devices                   |     External Devices     |
|:----------------------------------:|:------------------------:|
|       LIDAR                        |          Camera          |
|  Ultrasonic Unit                   | WIFI Connected Computers(brain) |
| Odometry(broken)                   |          Router(communication)          | -->



* Calibration: The courier has to run in a consistent coordinate system, which means the camera and lidar need to be calibrated to a global coordinate system. This process is simplified to align all coordinates to the predefined 2D image coordinate system. A robot coordinate is accessed by a chessboard fixed in the courier body. Because the odometry of the courier is broken, visual odometry plays an important role.

* Localization: Localization from the external camera is not accurate but enough for this scenario. A monocular camera has no scale estimation, although it can be approximated by cheeseboard grid size scale estimation. But it did not perform as thought. Finally, coordinates from the camera are simply triangulated with a fixed height parameter. Localization from LIDAR is initialized by coordinates from the camera but updated with local measurement of similarity transformation prediction via ICP algorithm. The location of the courier is determined by Kalman filtering.

* Mapping: It was not allowed to take any extra ROS packages in this project. I applied a 2-dimensional grid map with a moderate resolution, simply visualized in the OpenCV window, which can also be seen in the video. Besides, a grid map makes path planning easier.

* Motion Planning: A* algorithm with buffered/cost map.

* Control: Iteration of going and turning.

Techniques: C++, ROS, SLAM, Image Processing, LIDAR Processing.

### Dynamic Landmark based Visual Odometry(SFM)

![image-center]({{ "/assets/images/ezgif.com-optimize.gif" | relative_url }}){: .align-center style="width: 100%;"}

*Fig. 1 Visualization on Filtering of Matching points, below is the last frame, up is the current frame. Stereo images are aligned left and right. redpoints are removed.*
{: .text-center}

This is a classic project on SFM
* Extract feature points with multiple methods, such as SIFT, Harris, SURF, ORB, FREAK, BRISK, etc.
* Filter false matching points via stereo matching in the RANSAC framework.
* 3D motion estimation with frame-to-frame matching.
* Map and motion reconstruction, performance estimation(accuracy, efficiency), and model selection.

Techniques: C++, Visual Odometry, Scene Reconstruction.

### Object Tracking and Motion Prediction via UKF and EKF.

The scenario is the same as the last project, but this task is oriented to trajectory prediction preceding cars.
![image-center]({{ "/assets/images/ukf-highway-projected.gif" | relative_url }}){: .align-center}


*Fig. Visualization of prediction of prededing cars, source: [udacity](https://github.com/penguinflys/UdacitySensorFusion/tree/master/final_proj_uncented_kalman_filter_traffic_flow_tracking)*
{: .text-center}

Given: 
1. Car detection & tracking algorithm via neural network.
2. 3D active shape model(ASM) approximation method.

Task: 
* Filtering of point cloud and feature point via predicted bounding box
* Point cloud clustering and dense matched feature points clustering.
* ASM matching and object localization, and object matching.
* Kalman Filtering(KF) vs. EFK vs. UFK w.r.t efficiency and accuracy.

Techniques: Python, Object Tracking, Kalman Filtering, Motion Estimation.

### Trajectory Estimation with GPS + IMU based on Set-membership Kalman Filtering

![image-center]({{ "/assets/images/ikg_s4.jpg" | relative_url }}){: .align-center style="width: 100%;"}

*Fig. 3 Visualization of predicted ellipsoids, the colored patch are the ellipsoids. The probability distribution is not visualized*
{: .text-center}

Based on Paper: [Geo-Referencing of a Multi-Sensor System Based on Set-Membership Kalman Filter](https://www.researchgate.net/publication/327489443_Geo-Referencing_of_a_Multi-Sensor_System_Based_on_Set-Membership_Kalman_Filter).

The car is equipped with GPS and IMU sensor. In the conventional filtering method, the object is seen as a rigid body, and thus motion estimation is reduced as similarity transformation. However, non-rigid objects also need motion estimation, such as fluids in a ballon, which transforms itself when the water pressure changes. Uncertainty is seen as **ellipsoid space** surrounded by the probabilistic distribution. The application here is seen as a generalization test on a typical filtering scenario.

Techniques: Filtering

### Real-time HD Map Calibration with Multiple Lidar

![LIDARMAP]({{ "/assets/images/lidarmap.gif" | relative_url }}){: .align-center style="width: 100%;"}

* Prepare the measurement setup of 2 Lidar and a GPS receiver in a platform with a given calibration configuration. GPS could be used for localization, but here to synchronize the time of devices.
* Code a ROS package to input Velodyne-64 and Velodyne-16 raw data, then output calibrated point cloud to map to real-world, car pose is given by Mobile-mapping-system, which has a super high precision on localization.

Techniques: C++, ROS, Sensor Calibration.

### PanUrban Dataset - a panoptic dataset of aerial images([link]({{"assets/files/Master_Thesis_Presentation.pdf" | relative_url}}))

PanUrban Dataset is a dataset which takes car and building as things and trees impervious surface etc. as stuff, locating on the city region of Vaihingen and Potsdam. These datasets will be given to the public in a short time.

| Vaihingen sample |  Potsdam sample |
:-------------------------:|:-------------------------:
![Vaihingen]({{"assets/images/DenseBuilding_Potsdam.png" | relative_url}})  |  ![Potsdam]({{"assets/images/DenseBuilding_Vaihingen.png"| relative_url}})

Fig. Blue footprint encloses building instance, yellow footprint encloses car instance.

| Apartment | Factory | Innercity | Parking | Residual |
|:---------:|:---------:|:--------:|:-------:|:-------:|
|![img]({{"assets/images/samples/apartment.jpg" | relative_url }})|![img]({{"assets/images/samples/factory.jpg" | relative_url }})|![img]({{"assets/images/samples/innercity.jpg" | relative_url }})|![img]({{"assets/images/samples/parking.jpg" | relative_url }})|![img]({{"assets/images/samples/residual.jpg" | relative_url }})|
|![img]({{"assets/images/samples/apartment.png" | relative_url }})|![img]({{"assets/images/samples/factory.png" | relative_url }})|![img]({{"assets/images/samples/innercity.png" | relative_url }})|![img]({{"assets/images/samples/parking.png" | relative_url }})|![img]({{"assets/images/samples/residual.png" | relative_url }})|

Fig. Samples cross different city areas.

This dataset has the following properties:

* **Orthophoto**: Aerial Image dataset based on **orthogonal**([link](https://en.wikipedia.org/wiki/Orthophoto)) images with geospatial information, which can be directly used on measurement.
* **Multiple Tasks**: allows task for object detection, instance segmentation, semantic segmentation, and panoptic segmentation.
* **Adjacent Buildings**: unlike some datasets such as crowdAI, most buildings in our dataset are adjacent to their neighbors. In other words, it is __dense distributed__. thanks to the development of Instance Segmentation, the task to distinguish connected buildings is now possible.
* **Full Range Augmentation**: utilize features across source blocks to extract more robust features.

### Panoptic Segmentation on PanUrban dataset

Take [PanopticFPN](https://arxiv.org/abs/1901.02446) as an example model on PanUrban Dataset, with either rotated bounding box or axis-aligned bounding box.


| Panoptic Result | Semantic Label | Instance Label | Source Image |
|:-----------------:|:----------------:|:----------------:|:--------------:|
|![img]({{"assets/images/pred_samples/resarea2_Hameln0_pred.jpg" | relative_url }})|![img]({{"assets/images/pred_samples/resarea2_Hameln0_sem.png" | relative_url }})|![img]({{"assets/images/pred_samples/resarea2_Hameln0_gt.jpg" | relative_url }})|![img]({{"assets/images/pred_samples/resarea2_Hameln0_src.jpg" | relative_url }})|
|![img]({{"assets/images/pred_samples/top_potsdam_7_8_IRRG4_pred.jpg" | relative_url }}) |![img]({{"assets/images/pred_samples/top_potsdam_7_8_IRRG4_sem.png" | relative_url }})|![img]({{"assets/images/pred_samples/top_potsdam_7_8_IRRG4_gt.jpg" | relative_url }})|![img]({{"assets/images/pred_samples/top_potsdam_7_8_IRRG4_src.png" | relative_url }})|

*Fig. Visualization of PanUrban dataset Prediction*
{: .text-center}