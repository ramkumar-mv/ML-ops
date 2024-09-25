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
import concurrent.futures
from collections import defaultdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils.general import (LOGGER, check_img_size, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import non_max_suppression
import traceback

DETECTION_INTERVAL = int(os.getenv('DETECTION_INTERVAL', '5'))
SAVED_PLATES_DIR = os.getenv('SAVED_PLATES_DIR', '/app/saved_plates')
CLEAN_PLATES_DIR = os.getenv('CLEAN_PLATES_DIR', '/app/clean_plates')
CAMERA1 = os.getenv('CAMERA1', '')
CAMERA2 = os.getenv('CAMERA2', '')
WEIGHTS = os.getenv('WEIGHTS', "/app/best.pt")
DATA = os.getenv('DATA', "/app/Vehicle-Registration-Plates-2/data.yaml")

saved_plates_hashes = set()

os.makedirs(SAVED_PLATES_DIR, exist_ok=True)
os.makedirs(CLEAN_PLATES_DIR, exist_ok=True)

print(f"Monitoring directory: {SAVED_PLATES_DIR}")
print(f"Saving processed images to: {CLEAN_PLATES_DIR}")
print(f"Detection interval: {DETECTION_INTERVAL} seconds")
sys.stdout.flush()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

print("Initializing YOLO model...")
sys.stdout.flush()

device = select_device('')
model = DetectMultiBackend(WEIGHTS, device=device, data=DATA)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)
model.warmup(imgsz=(1, 3, *imgsz))

print("YOLO model initialized successfully")
sys.stdout.flush()

print("Loading OCR model...")
sys.stdout.flush()

# Load OCR model
ocr_model = tf.keras.models.load_model("/app/Fresh-Skew-CNN.h5")
print('OCR Model Loaded Successfully')
sys.stdout.flush()

# Precompute class labels for OCR
class_labels = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

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

    print(f"Processing image: {file_path}")
    sys.stdout.flush()

    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to read image: {file_path}")
        sys.stdout.flush()
        os.remove(file_path)
        return

    segmented_chars = segment_characters(file_path)
    if len(segmented_chars) < 9 or len(segmented_chars) > 10:
        print(f"Invalid number of characters detected: {len(segmented_chars)}")
        sys.stdout.flush()
        os.remove(file_path)
        return

    predicted_string = []

    for char_img in segmented_chars:
        img_array = load_and_preprocess_image(char_img)
        prediction = ocr_model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]
        predicted_string.append(predicted_label)
    
    predicted_string = ''.join(predicted_string)

    #print(f"Initial OCR Result: {predicted_string}")
    #sys.stdout.flush()

    if not is_valid_state_code(predicted_string):
        #print(f"Invalid state code detected: {predicted_string[:2]}. Discarding result.")
        #sys.stdout.flush()
        os.remove(file_path)
        return

    if not is_valid_format(predicted_string):
        #print(f"Invalid plate format: {predicted_string}")
        #sys.stdout.flush()
        os.remove(file_path)
        return

    current_time = time.time()
    last_detection_time = seen_strings.get(predicted_string, 0)

    if current_time - last_detection_time > DETECTION_INTERVAL:
        seen_strings[predicted_string] = current_time
        output_path = os.path.join(output_dir, f"{predicted_string}_{int(current_time)}.jpg")
        cv2.imwrite(output_path, image)
        detected_plates.add(file_path)
        
        print(f"Detected Valid Number Plate: {predicted_string}")
        sys.stdout.flush()
    
    os.remove(file_path)

def insert_plate_into_db(plate_number, image_name):
    LOGGER.info(f'platenumber = {plate_number}, image name ; {image_name}')
    '''try:
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
            connection.close()'''

@smart_inference_mode()
def run_detection(weights, sources, data, imgsz=(320, 320), conf_thres=0.5, iou_thres=0.6, max_det=1000, device='', save_crop=True):
    print(f"Starting detection with sources: {sources}")
    sys.stdout.flush()
    
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    datasets = []
    for source in sources:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride) if source.startswith('rtsp') else LoadImages(source, img_size=imgsz, stride=stride)
        datasets.append(dataset)
    
    model.warmup(imgsz=(1 if pt else len(datasets[0]), 3, *imgsz))
    frame_counts = [0] * len(datasets)

    print("Entering main detection loop")
    sys.stdout.flush()

    try:
        while True:
            for i, dataset in enumerate(datasets):
                for path, im, im0s, vid_cap, s in dataset:
                    frame_counts[i] += 1

                    if frame_counts[i] % 100 == 0:
                        print(f"Processed {frame_counts[i]} frames from source {i+1}")
                        sys.stdout.flush()

                    im = torch.from_numpy(im).to(model.device).half() if model.fp16 else torch.from_numpy(im).to(model.device).float()
                    im /= 255.0
                    if len(im.shape) == 3:
                        im = im[None]
                    
                    pred = model(im, augment=False)[0]
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, max_det=max_det)

                    for j, det in enumerate(pred):
                        im0 = im0s[j].copy() if isinstance(im0s, list) else im0s

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
                                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                                    save_path = Path(SAVED_PLATES_DIR) / f'plate_cam{i+1}_{frame_counts[i]}_{timestamp}.jpg'
                                    print(f'Saving cropped plate to {save_path}')
                                    Image.fromarray(crop).save(save_path)
                                    sys.stdout.flush()

    except Exception as e:
        print(f"An error occurred in the detection loop: {str(e)}")
        traceback.print_exc()
        sys.stdout.flush()


def process_saved_plates():
    print("Starting process_saved_plates function")
    sys.stdout.flush()
    
    event_handler = ImageHandler(SAVED_PLATES_DIR, CLEAN_PLATES_DIR, defaultdict(lambda: 0), set())
    observer = Observer()
    observer.schedule(event_handler, path=SAVED_PLATES_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def main():
    sources = [CAMERA1, CAMERA2] if CAMERA2 else [CAMERA1]
    sources = [s for s in sources if s]  # Remove any empty strings
    
    if not sources:
        print("No RTSP sources provided. Please set CAMERA1 and optionally CAMERA2 environment variables.")
        sys.exit(1)
    
    print(f"Starting detection process with sources: {sources}")
    sys.stdout.flush()

    detection_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    detection_future = detection_thread.submit(run_detection, WEIGHTS, sources, DATA)
    
    try:
        process_saved_plates()
    except KeyboardInterrupt:
        print("Stopping ANPR system...")
        sys.stdout.flush()
    except Exception as e:
        print(f"An error occurred in the main process: {str(e)}")
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        detection_thread.shutdown()
    
    print("ANPR system stopped.")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
