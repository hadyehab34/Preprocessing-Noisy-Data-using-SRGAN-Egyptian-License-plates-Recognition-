import torch
import numpy as np

import RRDBNet_arch as arch
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import os
import operator
from PIL import Image, ImageFilter
from flask import Flask, request, jsonify, render_template
import sys
import base64
import io
from base64 import b64encode

ENHANCE_MODEL_PATH = os.environ.get('ENHANCE_MODEL_PATH')
PD_MODEL_PATH = os.environ.get('PD_MODEL_PATH')
OCR_MODEL_PATH = os.environ.get('OCR_MODEL_PATH')
OCR_LABEL_MAP_PATH = os.environ.get('OCR_LABEL_MAP_PATH')
PD_LABEL_MAP_PATH = os.environ.get('PD_LABEL_MAP_PATH')
PD_PIPELINE_CONFIG_PATH = os.environ.get('PD_PIPELINE_CONFIG_PATH')
OCR_PIPELINE_CONFIG_PATH = os.environ.get('OCR_PIPELINE_CONFIG_PATH')


model_path1 = 'RRDB_PSNR_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
#device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
device = torch.device('cpu')

model1 = arch.RRDBNet(3, 3, 64, 23, gc=32)
model1.load_state_dict(torch.load(ENHANCE_MODEL_PATH), strict=True)
model1.eval()
PSNR = model1.to(device)

configs = config_util.get_configs_from_pipeline_file(PD_PIPELINE_CONFIG_PATH)
plate_detection = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=plate_detection)
ckpt.restore(PD_MODEL_PATH).expect_partial()


configs_ocr = config_util.get_configs_from_pipeline_file(OCR_PIPELINE_CONFIG_PATH)
ocr = model_builder.build(model_config=configs_ocr['model'], is_training=False)
ckpt_ocr = tf.compat.v2.train.Checkpoint(model=ocr)
ckpt_ocr.restore(OCR_MODEL_PATH).expect_partial()

def arrange(ind, classes, category_index):

    class_names_id_sorted = []
    
    if len(ind) ==0:
        s=""
    else:
        if type(operator.itemgetter(*ind)(classes)) is np.int64:
            class_id_detect_box = []
            class_id_detect_box.append(operator.itemgetter(*ind)(classes))
        else:
            class_id_detect_box = operator.itemgetter(*ind)(classes)
    for i in range(0, len(ind)):
        class_names_id_sorted.append(category_index[class_id_detect_box[i]]['name'])
    return class_names_id_sorted

def map_en_to_ar(a):
    b = {
    "1": "١",
    "2": "٢",
    "3": "٣",
    "4": "٤",
    "5": "٥",
    "6": "٦",
    "7": "٧",
    "8": "٨",
    "9": "٩",
    'a':"أ",
    'b':"ب",
    'd':"د",
    't':"ط",
    'v':"ض",
    'r':"ر",
    'k':"ق",
    'n':"ن",
    'h':"ه",
    'x':"ع",
    'y':"ى",
    'w':"و",
    's':"س",
    'j':"ص",
    'f':"ف",
    'g':"ج",
    'm':"م",
    'l':"ل"
    }
    s = " ".join(a)
    splitlines = s.split()
    for word in splitlines:
        if word in b:
            s = s.replace(word,str(b[word]))
    return s[::-1]

def box_vis(detections, num_detections, image):
    # create a dictionery from the items returns from object detection model.
    detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    detection_threshold = 0.5
    label_id_offset = 1
    category_index = label_map_util.create_category_index_from_labelmap(PD_LABEL_MAP_PATH)
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,
            line_thickness=7,
            category_index = category_index,
            min_score_thresh=detection_threshold,
            agnostic_mode=False)
    return image

def plate(detections, num_detections, image):
    # create a dictionery from the items returns from object detection model.
    detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    detection_threshold = 0.5
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    width = image.shape[1]
    height = image.shape[0]
    if len(boxes) == 0:
        region = 0
    else:
        for idx, box in enumerate(boxes):
            roi = box*[height, width, height, width]
            region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
    return region

def detect_fn(model, image):
    image = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    num_detections = int(detections.pop('num_detections'))
    l = [detections, num_detections]
    return l


def enhancing(model, region):
    region = region * 1.0 / 255
    region = torch.from_numpy(np.transpose(region[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = region.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    return output

def ocr_fn(model, image):
    image = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32) # convert the image to tensor to prepare it for object detection model.
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    num_detections = int(detections.pop('num_detections'))
    # create a dictionery from the items returns from object detection model.
    detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    detection_threshold = 0.4
    category_index_ocr = label_map_util.create_category_index_from_labelmap(OCR_LABEL_MAP_PATH)

    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes']
    
    scores_test = np.squeeze(detections['detection_scores'])
    bboxes = boxes[scores_test > detection_threshold]
    classes = detections['detection_classes'][:len(scores)]+1
    
    ind=np.argsort(bboxes[:,1])
    l = arrange(ind, classes, category_index_ocr)
    s = map_en_to_ar(l)
    return s


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/EEALPR', methods=['POST'])
def detect_objects_route():
    # Get the image file from the request
    image = request.files['image'].read()

    # Convert the image to a NumPy array
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect the objects in the image
    l = detect_fn(plate_detection, image)
    detections = l[0]
    num_detections = l[1]
    region = plate(detections,num_detections, image)
    image_np_with_detections = box_vis(detections,num_detections, image)
    im = Image.fromarray(image_np_with_detections)
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    if type(region) != type(np.array([0])):
        text = ""
    else:
        enhanced_img = enhancing(PSNR, region)
        text = ocr_fn(ocr, enhanced_img)
    # Return the detection results as a JSON response
    return render_template('index.html', ocr = text, img_data=encoded_img_data.decode('utf-8'))

if __name__ == '__main__':
    app.run(port=5000,debug=True)