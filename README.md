# Car Detector

This project implements a car detection system using the YOLO (You Only Look Once) object detection algorithm and the SORT (Simple Online and Realtime Tracking) algorithm for object tracking. It processes video input to identify and track vehicles and can be configured to use a mask and tracking lines.
This repository will help you to build your own custom car detector. In this repo, I have take a particular camera position and created mask and tracking line for that particular camera position. Developer will need to update the tracking line, limits variable and the mask for the region in which they want to detect the cars.

## Installation

This section describes how to set up and run the Car Detector application, including environment setup and dependency installation.

### Prerequisites

- Python 3.6+
- Ensure all dependencies are installed from the `requirements.txt` file, which includes libraries like OpenCV, Ultralytics' YOLO, and others necessary for video processing and object detection.

### Setup

1. **Clone this repository** or download the source code.
   
   ```bash
   git clone https://github.com/Niticodersh/Car-Detector-using-YOLOv8.git
   cd Car-Detector-using-YOLOv8
   ```
2. **Create a virtual environment** 
- For Unix/Linux/MacOS:
   ```bash
   pyhton3 -m venv detector_venv
   source detector_venv/bin/activate
   ```
- For Windows:
   ```bash
   python -m venv detector_venv
   .\detector_venv\Scripts\activate
   ```
3. **Install the requirements from yhe `requirements.txt` file**

   ```bash
   pip install -r requirements.txt
   ```
4. **Usage**
 To run the Car Detector, use the following command:
 ```bash
  python car_detector.py --video-path <path_to_video> --mask-path <path_to_mask> --tracking-line <x1 y1 x2 y2>
```
Parameters

--video-path - Path to the video file you want to process. 

--mask-path - Optional. Path to the mask image file to apply to the video.

--tracking-line - Coordinates to set the tracking line (format: x1 y1 x2 y2).

Features

Object detection using YOLOv8.

Object tracking using SORT.

Ability to apply a mask to the video.

Configurable tracking lines for counting vehicles.

Output
The system outputs a video stream that shows detected and tracked vehicles, with vehicle counts displayed on the screen.

## Car Detection Results

Here is the output of the Car Detector using YOLOv8:

![Car Detector Results](https://github.com/Niticodersh/Car-Detector-using-YOLOv8/blob/main/results.png?raw=true)

6. **Contributing**
Contributions to this project are welcome. Please fork the repository and submit a pull request.

7. **Authors**
Nitish Bhardwaj

8. **Acknowledgments**
-Thanks to the ultralytics team for the YOLO implementation.
-Thanks to Alex Bewley for the SORT algorithm.

9. Contact
For issues, questions, or professional inquiries, please contact [bhardwaj.11@iitj.ac.in].
=======
