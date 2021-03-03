---
permalink: /about/
title: "About Me"
layout: single
header:
  video:
    width: 70%
    id: 9dfWzp7rYR4
    provider: youtube
toc: true
toc_sticky: true
comments: true
author_profile: true
---
## Biography

Hello 🎅🏼, my name is Jie Yuan. Welcome to my site.

I am a graduate of the [Leibiniz Universität Hannover](https://www.uni-hannover.de/) program [Navigation and Field Robotics](https://www.uni-hannover.de/en/studium/studienangebot/info/studiengang/detail/navigation-and-field-robotics/) specializing in deep learning(DL) application in mobile robotics and Simultaneous Localization and Mapping(SLAM). Python(for deep learning) and C++(for SLAM) are my principal developing languages, and sometimes I use Matlab for verification of algorithms.

<!-- I used to be dogmatic, selfish, and ambitious to become someone that can make a difference, but God has changed me to love others take care of people whom I love and give more instead of asking more. In the aspect of personality and psychology, I am a bit like "Bruce" in [7 Up series](https://en.wikipedia.org/wiki/Up_(film_series)). -->

If you want to know whether a penguin can really fly or not, the answer is NO, definitely! Video in the head might be the greatest Fools pranks of all time of BBC.

Please feel free to contact me [📧](mailto:yuanjielovejesus@gmail.com).

## Skills

I am fond of cooking👩‍🍳🍚🏺🥠. If it is possible to select a second occupation, it must be a cook. I subscribe youtube channels [Chef Wang](https://www.youtube.com/channel/UCg0m_Ah8P_MQbnn77-vYnYw), [老饭骨](https://www.youtube.com/channel/UCBJmYv3Vf_tKcQr5_qmayXg), and [小高姐的 Magic Ingredients](https://www.youtube.com/channel/UCCKlp1JI9Yg3-cUjKPdD3mw).


## Technique Books

Here are books 📚 that I like reviews a lot during development:

* [Probabilistic Robotics](https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf)
* [State Estimation for Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf)
* [Multiple View Geometry in Computer Vision](https://www.amazon.com/Multiple-View-Geometry-Computer-Vision/dp/0521540518)
* [Deep Learning](https://www.deeplearningbook.org/)
* [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)


## Projects

The following are some projects that is done as student or student research assistant. 

### Digital Earth based on WMS [link](https://github.com/penguinflys/Oriental_EYE)

![image-center]({{ "/assets/images/digitalearth.png" | relative_url }}){: .align-center style="width: 100%;"}
*Fig. 0 Visualization of Digital Earth with DEM model*
{: .text-center}

This project is a teaching project which simulates the elevation of earth, and depth of the ocean, it is initially based on the [icosahedron](https://en.wikipedia.org/wiki/Icosahedron), and iteratively refined in a deeper level to sample the local elevation. it can also simulate the gravity field if the data is in hand.

### LEGO Courier Student Toy Project

{% include video id="Rj_TkF2gSKw" provider="youtube" %}

This is the first project that requires knowledge of SLAM. it is developed in C++ with ROS as a communication tool, PCL and OpenCV as data processing API. It is conducted in a closed arena with lots of poles and a short tunnel. The examiner set the arena scenario, delivering destination and courier born place randomly, which requires the robustness of the method. What I developed is in the last of the video, namely SLAM method. The development can be mainly divided into the following pieces.

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



* Calibration: The courier has to run in a consistent coordinate system, which means camera and lidar need to be calibrated to global coordinate system. This process is simplified to align all coordinates to the predefined 2D image coordinate system. Robot coordiante is accesed by a cheesboard fixed in the courier body. LIDAR is calibrated with manual tape measurement. Because the odometry of courier is broken, visual odometry plays an important role.

* Localization: Localization from the external camera is not accurate but enough for this scenario. As a monocular camera has no scale estimation, although it can be approximated by cheeseboard grid size scale estimation. But it did not perform as thought. Finally, coordinate from camera are simply triangulated with a fixed height parameter. Localization from LIDAR is initialized by coordinates from camera, but updated with local measurement of similarity transformation prediction via ICP algorithm. The location of the courier is determined by Kalman filtering.

* Mapping: It was not allowed to take any extra ROS packages in this project. I applied 2-dimensional grid map, with a moderate resolution, simply visualized in the OpenCV window, which can also be seen in the video. Besides, a grid map makes path planning easier.

* Motion Planning: A* algorithm with buffered/cost map.

* Control: Iteration of going and turning.








### Dynamic Landmark based Visual Odometry(SFM)

![image-center]({{ "/assets/images/ezgif.com-optimize.gif" | relative_url }}){: .align-center style="width: 100%;"}

*Fig. 1 Visualization on Filtering of Matching points, below is last frame, up is current frame. stero images are aligned left and right. red point are removed points*
{: .text-center}

This is a classic project, 
* Extreact feature points with multiple methods, such as SIFT, Harris, SURF, ORB,FREAK, BRISK, etc.
* Filter false matching points via stero matching in RANSAC framework.
* 3D motion estimation with frame-to-frame matching.
* Map and motion reconstruction, performance estimation(accuracy, efficiency) and model selection.

### Object Tracking and Motion Prediction via KFs.

Senerio is the same as the last project, but this task is oriented to trajectory prediction of preciding cars.
![image-center]({{ "/assets/images/ukf-highway-projected.gif" | relative_url }}){: .align-center}


*Fig. 2 Visualization of prediction of prededing cars, source: [udacity](https://github.com/penguinflys/UdacitySensorFusion/tree/master/final_proj_uncented_kalman_filter_traffic_flow_tracking)*
{: .text-center}

Given: 
1. Car detection & tracking algorithm via neural network.
2. 3D active shape model(ASM) approximation method.

Task: 
* Filtering of point cloud and feature point via predicted bounding box
* Point cloud clustering and dense matched feature points clustering.
* ASM matching and object localization, and object matching.
* Kalman Filtering(KF) vs EFK vs UFK w.r.t efficiency and accuracy.

### Trajectory Estimation with GPS + IMU based on Set-membership Kalman Filtering

![image-center]({{ "/assets/images/ikg_s4.jpg" | relative_url }}){: .align-center style="width: 100%;"}

*Fig. 3 Visualization of predicted elipsoids, the colored patch are the elipsoids. Probability distribution is not visualized*
{: .text-center}

Based on Paper: [Geo-Referencing of a Multi-Sensor System Based on Set-Membership Kalman Filter](https://www.researchgate.net/publication/327489443_Geo-Referencing_of_a_Multi-Sensor_System_Based_on_Set-Membership_Kalman_Filter).

Trajectory optimization of a car equipped with GPS and IMU sensor. In the convemtional filtering method, object is seen as a rigid body, and thus motion estimation are reduced as similarity transformation. However non-rigid objects also need motion estimation such as fluids in ballon, which trasforms itself when water pressure changes. Uncertainty is seen as **ellipsoid space** surrounded with probabilistic distribution. The application here are seen as a generalization test on normal filtering seneriao.


### Real-time HD Map Calibration with Multiple Lidars

![LIDARMAP]({{ "/assets/images/lidarmap.gif" | relative_url }}){: .align-center style="width: 100%;"}

* Prepare the measurement setup of 2 Lidars and a GPS receiver in a platform with given calibration configuration. GPS could be used to localization but here to synchronize time of devices.
* Code a ROS package to input velodyne-64 and velodyne-16 raw data, then output calibrated point cloud to map to real-world, car pose is given by Mobile-mapping-system, which has a super high precision on localization.

