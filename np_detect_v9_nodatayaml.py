import os
import time
import sys
from pathlib import Path
import torch
import cv2
import imagehash
from PIL import Image
import numpy as np
import tensorflow as tf
import re
import mysql.connector
import concurrent.futures
from collections import defaultdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils.general import (LOGGER, Profile, check_img_size, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import non_max_suppression
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize hash-based storage to filter out duplicate or similar detections
saved_plates_hashes = set()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

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
            
            time.sleep(0.1)

def filter_by_size(xyxy, img_shape):
    MIN_WIDTH, MIN_HEIGHT = 50, 20
    MAX_WIDTH, MAX_HEIGHT = img_shape[1] * 0.5, img_shape[0] * 0.2

    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    
    return MIN_WIDTH <= width <= MAX_WIDTH and MIN_HEIGHT <= height <= MAX_HEIGHT

@smart_inference_mode()
def run_detection(
        weights,
        source,
        imgsz=(320, 320),
        conf_thres=0.5,
        iou_thres=0.6,
        max_det=1000,
        device='',
        save_crop=True,
        nosave=True,
        classes=None,
        agnostic_nms=False,
        augment=False,
        half=False,
        vid_stride=2,
):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadStreams(source, img_size=imgsz, stride=stride) if source.startswith('rtsp') else LoadImages(source, img_size=imgsz, stride=stride)
    
    model.warmup(imgsz=(1 if pt else len(dataset), 3, *imgsz))
    frame_count = 0

    for path, im, im0s, vid_cap, s in dataset:
        frame_count += 1

        im = torch.from_numpy(im).to(model.device).half() if model.fp16 else torch.from_numpy(im).to(model.device).float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]
        
        pred = model(im, augment=augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy() if isinstance(im0s, list) else im0s

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if not filter_by_size(xyxy, im0.shape):
                        continue

                    crop = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    crop_hash = imagehash.average_hash(Image.fromarray(crop))
                    if crop_hash in saved_plates_hashes:
                        continue
                    saved_plates_hashes.add(crop_hash)

                    if save_crop:
                        save_dir = config['Paths']['SavedPlatesDir']
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        save_path = Path(save_dir) / f'plate_{frame_count}_{timestamp}.jpg'
                        print(f'Saving cropped plate to {save_path}')
                        Image.fromarray(crop).save(save_path)

    LOGGER.info(f'Done. Processed {frame_count} frames.')

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
        os.remove(file_path)
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
    detection_interval = int(config['Detection']['DetectionInterval'])

    if is_valid_format(predicted_string) and is_valid_state_code(predicted_string):
        if current_time - last_detection_time > detection_interval:
            seen_strings[predicted_string] = current_time
            output_path = os.path.join(output_dir, os.path.basename(file_path))
            cv2.imwrite(output_path, image)
            detected_plates.add(file_path)
            
            print(f"Detected Number Plate: {predicted_string}")
            insert_plate_into_db(predicted_string, os.path.basename(file_path))
            
            os.remove(file_path)
        else:
            os.remove(file_path)
    else:
        os.remove(file_path)

def insert_plate_into_db(plate_number, image_name):
    try:
        connection = mysql.connector.connect(
            host=config['Database']['Host'],
            user=config['Database']['User'],
            database=config['Database']['DatabaseName']
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

def main():
    weights = "/Users/ramkumarmv/Desktop/iocl-ml/main-code/home/resiliente63/Desktop/ml/yolov9/runs/train/exp/weights/best.pt"
    source = config['Detection']['RTSPLink']
    #data = ROOT / "/Users/ramkumarmv/Desktop/iocl-ml/main-code/Vehicle-Registration-Plates-2/data.yaml"
    
    input_dir = config['Paths']['SavedPlatesDir'] 
    output_dir = config['Paths']['CleanPlatesDir']
    
    seen_strings = defaultdict(lambda: 0)
    detected_plates = set()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Start the detection process in a separate thread
    detection_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    detection_future = detection_thread.submit(run_detection, weights, source)
    
    # Set up the image handler for OCR
    event_handler = ImageHandler(input_dir, output_dir, seen_strings, detected_plates)
    observer = Observer()
    observer.schedule(event_handler, path=input_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
            # Check if the detection process has finished
            if detection_future.done():
                print("Detection process has finished. Restarting...")
                detection_future = detection_thread.submit(run_detection, weights, source)
    except KeyboardInterrupt:
        observer.stop()
        detection_thread.shutdown()
    
    observer.join()

if __name__ == "__main__":
    # Load the OCR model
    model = tf.keras.models.load_model("/Users/ramkumarmv/Desktop/iocl-ml/Fresh-Skew-CNN.h5")
    print('OCR Model Loaded Successfully')
    main()