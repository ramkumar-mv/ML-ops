import os
import time
from pathlib import Path
import torch
import cv2
import imagehash
from PIL import Image
from utils.general import (LOGGER, Profile, check_img_size, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import non_max_suppression
import sys
import numpy as np
import tensorflow as tf
import re
import mysql.connector
import concurrent.futures
from collections import defaultdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Initialize hash-based storage to filter out duplicate or similar detections
saved_plates_hashes = set()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def filter_by_size(xyxy, img_shape):
    """Filter bounding boxes based on the expected size of number plates."""
    MIN_WIDTH, MIN_HEIGHT = 50, 20
    MAX_WIDTH, MAX_HEIGHT = img_shape[1] * 0.5, img_shape[0] * 0.2  # Relative to image size

    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    
    return MIN_WIDTH <= width <= MAX_WIDTH and MIN_HEIGHT <= height <= MAX_HEIGHT

@smart_inference_mode()
def run(
        weights='/Users/ramkumarmv/Desktop/iocl-ml/main-code/home/resiliente63/Desktop/ml/yolov9/runs/train/exp/weights/best.pt',  # path to the trained model
        source='rtsp://admin:msfconsole1%24@192.168.30.27:554/onvif1',  # RTSP link
        data=ROOT / '/Users/ramkumarmv/Desktop/iocl-ml/main-code/Vehicle-Registration-Plates-2/data.yaml',  # dataset.yaml path
        imgsz=(320, 320),  # inference size
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.6,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or cpu
        save_crop=True,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        vid_stride=2,  # video frame-rate stride
):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load video source
    dataset = LoadStreams(source, img_size=imgsz, stride=stride) if source.startswith('rtsp') else LoadImages(source, img_size=imgsz, stride=stride)
    
    # Load the OCR model
    ocr_model = tf.keras.models.load_model('/Users/ramkumarmv/Desktop/iocl-ml/Fresh-Skew-CNN.h5')
    print('OCR Model Loaded Successfully')

    # Run inference
    model.warmup(imgsz=(1 if pt else len(dataset), 3, *imgsz))  # warmup
    frame_count = 0  # Initialize frame count for saving images
    seen_strings = defaultdict(lambda: 0)
    detected_plates = set()

    for path, im, im0s, vid_cap, s in dataset:
        frame_count += 1  # Increment frame count for each loop iteration

        im = torch.from_numpy(im).to(model.device).half() if model.fp16 else torch.from_numpy(im).to(model.device).float()
        im /= 255.0  # Normalize
        if len(im.shape) == 3:
            im = im[None]  # add batch dimension
        
        # Inference
        pred = model(im, augment=augment)[0]
        
        # Non-Max Suppression
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # per image
            im0 = im0s[i].copy() if isinstance(im0s, list) else im0s

            if len(det):
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Process detections
                for *xyxy, conf, cls in reversed(det):
                    # Filter based on the bounding box size (skip too large/small)
                    if not filter_by_size(xyxy, im0.shape):
                        continue

                    # Hash the cropped region to avoid duplicate detections
                    crop = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    crop_hash = imagehash.average_hash(Image.fromarray(crop))
                    if crop_hash in saved_plates_hashes:
                        continue  # skip if duplicate
                    saved_plates_hashes.add(crop_hash)

                    # Process OCR directly
                    process_ocr_on_plate(crop, ocr_model, seen_strings, detected_plates)

def process_ocr_on_plate(cropped_plate, ocr_model, seen_strings, detected_plates):
    # Preprocess and segment the cropped plate image for OCR
    segmented_chars = segment_characters(cropped_plate)
    
    if len(segmented_chars) < 9 or len(segmented_chars) > 10:
        return  # Skip if character count is not valid

    class_labels = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]
    predicted_string = []

    # Perform OCR prediction
    for char_img in segmented_chars:
        img_array = load_and_preprocess_image(char_img)
        prediction = ocr_model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]
        predicted_string.append(predicted_label)
    
    predicted_string = listToString(predicted_string)

    current_time = time.time()
    last_detection_time = seen_strings.get(predicted_string, 0)

    if is_valid_format(predicted_string) and is_valid_state_code(predicted_string):
        if current_time - last_detection_time > 20:  # Check if 20 seconds have passed
            seen_strings[predicted_string] = current_time
            print(f"Detected Number Plate: {predicted_string}")
            # Optionally, insert detected plate into the database
            # insert_plate_into_db(predicted_string)
        else:
            print(f"Ignored duplicate detection of {predicted_string} within 20 seconds.")


class ImageHandler(FileSystemEventHandler):
    def __init__(self, input_dir, output_dir, seen_strings, detected_plates):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.seen_strings = seen_strings
        self.detected_plates = detected_plates

    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = event.src_path
            self.wait_for_file_complete(file_path)
            process_image(file_path, self.detected_plates, self.seen_strings, self.output_dir)

    def wait_for_file_complete(self, file_path, timeout=5):
        """Wait for the file to be completely written."""
        last_size = -1
        start_time = time.time()
        
        while True:
            current_size = os.path.getsize(file_path)
            if current_size == last_size:
                break
            last_size = current_size
            
            if time.time() - start_time > timeout:
                print(f"Timeout while waiting for {file_path} to finish writing.")
                break
            
            time.sleep(0.1)  # Wait a bit before checking again


def preprocess_image(image_path):
    image_data = np.fromfile(image_path, np.uint8)  # Read image file to numpy array
    image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)  # Decode to grayscale
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def segment_characters(image_path):
    thresh_image = preprocess_image(image_path)
    num_labels, labels = cv2.connectedComponents(thresh_image)
    total_pixels = thresh_image.shape[0] * thresh_image.shape[1]
    lower = total_pixels // 200
    upper = total_pixels // 30
    mask = np.zeros(thresh_image.shape, dtype="uint8")

    for label in range(1, num_labels):
        label_mask = np.zeros(thresh_image.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
        if lower < num_pixels < upper:
            mask = cv2.add(mask, label_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    character_images = []
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if w > 5 and h > 5 and 0.2 <= aspect_ratio <= 1.5:
            char_img = thresh_image[y:y+h, x:x+w]
            padding = 5
            char_img = cv2.copyMakeBorder(char_img, padding, padding, padding, padding,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])
            char_img = cv2.resize(char_img, (64, 64))
            char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)
            char_img = cv2.bitwise_not(char_img)
            character_images.append(char_img)
            bounding_boxes.append((x, y, w, h))

    if len(character_images) > 1:
        character_images, bounding_boxes = remove_outliers(character_images, bounding_boxes)

    if len(character_images) == 0:
        thresh_image = cv2.bitwise_not(thresh_image)
        character_images, bounding_boxes = segment_characters_alternative(thresh_image)

    return character_images

def remove_outliers(character_images, bounding_boxes):
    if len(character_images) <= 1:
        return character_images, bounding_boxes

    heights = [bb[3] for bb in bounding_boxes]
    y_positions = [bb[1] for bb in bounding_boxes]
    median_height = np.median(heights)
    median_y = np.median(y_positions)

    filtered_chars = []
    filtered_boxes = []
    for char, box in zip(character_images, bounding_boxes):
        _, y, _, h = box
        if 0.5 * median_height < h < 1.5 * median_height and abs(y - median_y) < median_height:
            filtered_chars.append(char)
            filtered_boxes.append(box)

    return filtered_chars, filtered_boxes

def segment_characters_alternative(thresh_image):
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    character_images = []
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        if w > 5 and h > 5 and 0.2 <= aspect_ratio <= 1.5:
            char_img = thresh_image[y:y+h, x:x+w]
            padding = 5
            char_img = cv2.copyMakeBorder(char_img, padding, padding, padding, padding,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])
            char_img = cv2.resize(char_img, (64, 64))
            char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)
            char_img = cv2.bitwise_not(char_img)
            character_images.append(char_img)
            bounding_boxes.append((x, y, w, h))

    return character_images, bounding_boxes

def load_and_preprocess_image(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def listToString(s):
    return ''.join(s)

def is_valid_format(predicted_string):
    pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')
    return bool(pattern.match(predicted_string))

def is_valid_state_code(predicted_string):
    state_codes = {"AR", "AS", "BR", "CG", "DL", "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LD", "MP", "MH", "MN",
                   "ML", "MZ", "NL", "OD", "PY", "PB", "RJ", "SK", "TN", "TS", "UP", "UK", "WB"}
    return predicted_string[:2] in state_codes

def process_image(file_path, detected_plates, seen_strings, output_dir):
    if file_path in detected_plates:
        return

    image = cv2.imread(file_path)
    if image is None:
        return

    segmented_chars = segment_characters(file_path)
    if len(segmented_chars) < 9 or len(segmented_chars) > 10:
        os.remove(file_path)  # Remove the image if it doesn't have the expected number of characters
        return

    class_labels = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]
    predicted_string = []

    for char_img in segmented_chars:
        img_array = load_and_preprocess_image(char_img)
        prediction = model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]
        predicted_string.append(predicted_label)
    
    predicted_string = listToString(predicted_string)

    current_time = time.time()
    last_detection_time = seen_strings.get(predicted_string, 0)

    if is_valid_format(predicted_string) and is_valid_state_code(predicted_string):
        if current_time - last_detection_time > 5:  # Check if 20 seconds have passed
            seen_strings[predicted_string] = current_time
            output_path = os.path.join(output_dir, os.path.basename(file_path))
            cv2.imwrite(output_path, image)
            detected_plates.add(file_path)
            
            print(f"Detected Number Plate: {predicted_string}")
            insert_plate_into_db(predicted_string, os.path.basename(file_path))
            
            os.remove(file_path)
        else:
            os.remove(file_path)  # Remove the image if it has been detected within 20 seconds
    else:
        os.remove(file_path)  # Remove the image if it's not valid

def insert_plate_into_db(plate_number, image_name):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            database='gemesh'
        )
        
        cursor = connection.cursor()
        sql_insert_query = """INSERT INTO detected_plates (plate_number, image_name) VALUES (%s, %s)"""
        record = (plate_number, image_name)
        cursor.execute(sql_insert_query, record)
        connection.commit()
        print(f"Inserted {plate_number} into the database")
        
    except mysql.connector.Error as error:
        print(f"Failed to insert record into database: {error}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def process_images(input_dir, output_dir):
    seen_strings = defaultdict(lambda: 0)  # Initialize with timestamp
    detected_plates = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(input_dir, filename)
                futures.append(executor.submit(process_image, file_path, detected_plates, seen_strings, output_dir))

        # Wait for all threads to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()

def main():
    input_dir = '/Users/ramkumarmv/Desktop/iocl-ml/main-code/home/resiliente63/Desktop/ml/yolov9/SAVEDPLATES1'
    output_dir = '/Users/ramkumarmv/Desktop/iocl-ml/main-code/home/resiliente63/Desktop/ml/yolov9/CLEANPLATES1'
    
    seen_strings = defaultdict(lambda: 0)  # Initialize with timestamp
    detected_plates = set()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    event_handler = ImageHandler(input_dir, output_dir, seen_strings, detected_plates)
    observer = Observer()
    observer.schedule(event_handler, path=input_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    run(
        weights='/Users/ramkumarmv/Desktop/iocl-ml/main-code/home/resiliente63/Desktop/ml/yolov9/runs/train/exp/weights/best.pt',
        source='rtsp://admin:msfconsole1%24@192.168.30.27:554/onvif1',
        half=False
    )
    model = tf.keras.models.load_model('/Users/ramkumarmv/Desktop/iocl-ml/Fresh-Skew-CNN.h5')
    print('Model Loaded Successfully')
    main()
