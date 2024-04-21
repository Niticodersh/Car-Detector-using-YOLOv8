from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import  *

import argparse
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Car Detector CLI")
    parser.add_argument("--video-path", type=str, help="Path to the video file")
    parser.add_argument("--mask-path", type=str, help="Path to the mask file")
    parser.add_argument("--tracking-line", type=str, help="Coordinates of the tracking line (x1 y1 x2 y2)")
    return parser.parse_args()

def get_input(prompt, required=True, validator=None):
    """General purpose input function with validation."""
    while True:
        user_input = input(prompt)
        if not user_input and required:
            logging.error("This input is required. Please try again.")
            continue
        if validator:
            try:
                if validator(user_input):
                    return user_input
                else:
                    logging.error("Invalid input. Please try again.")
            except ValueError as e:
                logging.error(f"Invalid input: {e}. Please try again.")
        else:
            return user_input

def validate_coordinates(input_str):
    """Validate that the input string can be converted to four integers."""
    coords = list(map(int, input_str.split()))
    if len(coords) != 4:
        raise ValueError("Exactly four integers are required.")
    return True

def main():
    args = parse_args()

    if not all(vars(args).values()):  # If any argument is missing
        logging.info("---------User Settings-----------")
        logging.info("*** If the file is in same directory, use relative path, else use absolute path ***")

        video_path = args.video_path or get_input("Enter the path of your video: ")
        mask_path = args.mask_path
        choice = get_input("Do you have mask to apply on original image: y/n ?  ")
        if choice == 'y':
            mask_path = get_input("Enter the path of your mask: ")
        else:
            mask_path = None
        input_limits = args.tracking_line or get_input("Enter the coordinates of your tracking line (x1 y1 x2 y2): ", validator=validate_coordinates)
        limits = list(map(int, input_limits.split()))
    else:
        video_path, mask_path, input_limits = args.video_path, args.mask_path, args.tracking_line
        limits = list(map(int, input_limits.split()))

    # Here you might continue with processing the video using the provided inputs
    logging.info("Video Path: %s", video_path)
    if mask_path:
        logging.info("Mask Path: %s", mask_path)
    logging.info("Tracking Line Coordinates: %s", limits)
    detector(video_path, mask_path, limits)


def detector(video_path, mask_path, limits):

    cap = cv2.VideoCapture(video_path) #For Video
    model = YOLO('../yolo-weights/yolov8n.pt')
    class_names = {0: u'__background__',
     1: u'person',
     2: u'bicycle',
     3: u'car',
     4: u'motorcycle',
     5: u'airplane',
     6: u'bus',
     7: u'train',
     8: u'truck',
     9: u'boat',
     10: u'traffic light',
     11: u'fire hydrant',
     12: u'stop sign',
     13: u'parking meter',
     14: u'bench',
     15: u'bird',
     16: u'cat',
     17: u'dog',
     18: u'horse',
     19: u'sheep',
     20: u'cow',
     21: u'elephant',
     22: u'bear',
     23: u'zebra',
     24: u'giraffe',
     25: u'backpack',
     26: u'umbrella',
     27: u'handbag',
     28: u'tie',
     29: u'suitcase',
     30: u'frisbee',
     31: u'skis',
     32: u'snowboard',
     33: u'sports ball',
     34: u'kite',
     35: u'baseball bat',
     36: u'baseball glove',
     37: u'skateboard',
     38: u'surfboard',
     39: u'tennis racket',
     40: u'bottle',
     41: u'wine glass',
     42: u'cup',
     43: u'fork',
     44: u'knife',
     45: u'spoon',
     46: u'bowl',
     47: u'banana',
     48: u'apple',
     49: u'sandwich',
     50: u'orange',
     51: u'broccoli',
     52: u'carrot',
     53: u'hot dog',
     54: u'pizza',
     55: u'donut',
     56: u'cake',
     57: u'chair',
     58: u'couch',
     59: u'potted plant',
     60: u'bed',
     61: u'dining table',
     62: u'toilet',
     63: u'tv',
     64: u'laptop',
     65: u'mouse',
     66: u'remote',
     67: u'keyboard',
     68: u'cell phone',
     69: u'microwave',
     70: u'oven',
     71: u'toaster',
     72: u'sink',
     73: u'refrigerator',
     74: u'book',
     75: u'clock',
     76: u'vase',
     77: u'scissors',
     78: u'teddy bear',
     79: u'hair drier',
     80: u'toothbrush'}

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    # limits = [340, 380, 980, 380]
    totalCounts=[]
    while True:
        success, img = cap.read()
        if img is None:
            break
        roi = img
        if mask_path != None:
            mask = cv2.imread(mask_path)
            # print(img.shape, mask.shape)
            if img is not None:
                if img.shape[:2] != mask.shape [:2]:
                 mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                 roi = cv2.bitwise_and(img, mask)

        results = model(roi, stream=True)

        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 0), 3)
                bbox = int(x1), int(y1), int(x2-x1), int(y2-y1)

                # Confidence Score
                conf = math.ceil((box.conf[0]*100))/100

                #Class Name
                cls = box.cls[0]

                current_class = class_names[int(cls)+1]

                if current_class == 'car' or current_class == 'truck' or current_class == 'bus'\
                        or current_class == 'motorbike' and conf > 0.2:
                    # cvzone.cornerRect(img, bbox, l=9, rt=5)
                    # cvzone.putTextRect(img, f'{current_class} ,{conf}', (max(0, x1), max(35, y1)), scale=0.7,  thickness=1, offset=3)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(2555,0,255))
            cvzone.putTextRect(img, f'id:{int(id)}', (max(0, x1), max(35, y1)), scale=1,  thickness=1, offset=3)
            cx, cy = x1+w//2, y1+w//2
            cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)
            # print(f'{limits[0]} < {cx} < {limits[2]} and {limits[1] - 15} < {cy} <  {limits[3] + 15}')
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy <  limits[3] + 15:
                if totalCounts.count(id) == 0:
                    totalCounts.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        cvzone.putTextRect(img, f'Count: {len(totalCounts)}', (50, 50))
        # cv2.imshow("Results", img)
        # cv2.imshow("ROI", roi)
        # Define a scale factor
        scale_factor = 1  # Adjust this based on your screen size

        if img is not None:
            # Resize the image
            display_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            display_roi = cv2.resize(roi, (0, 0), fx=scale_factor, fy=scale_factor)

            # Display the resized images
            cv2.imshow("Results", display_img)
        # cv2.imshow("ROI", display_roi)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()