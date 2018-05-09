from keras.models import load_model
from PIL import Image
from itertools import product
import numpy as np
from yad2k_out import get_model
import datetime as dt

# Defaults
ANCHORS_PATH = "model/yolov2.anchors"
CLASSES_PATH = "model/yolov2.classes"
CONFIG_PATH  = "model/yolov2.cfg"
WEIGHTS_PATH = "model/yolov2.weights" 
MODEL_IMAGE_SIZE = (608, 608)
MIN_CONFIDENCE = 0.5
MAX_OVERLAP = 0.35

model = None

def get_classes(classes_path):
    """ Loads classes from text file 
        Input: 
            classes_path (string) - The path of classes
        Output:
            class_names (list) - Class names in list format
            class_mappings(dict) - Dictionary mapping class names with id e..g {"aeroplan": 4, "apple": 47 ...} 
    """
    with open(classes_path) as f:
        class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        class_mappings = dict(zip(class_names, range(len(class_names))))
        return class_names, class_mappings

def get_anchors(anchors_path):
    """ Loads anchors from a comma separated text file
            Input: 
                anchors_path (string) - The path of anchors
            Output:
                anchors (numpy array) - The anchors formated to [n, 2], where n is the number of anchors
        """
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [np.float32(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), -1)

def iou(a, b):
    """ Calculates intersection over union (IOU) over two tuples """
    (a_x1, a_y1), (a_x2, a_y2) = a
    (b_x1, b_y1), (b_x2, b_y2) = b
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    
    dx = min(a_x2, b_x2) - max(a_x1, b_x1)
    dy = min(a_y2, b_y2) - max(a_y1, b_y1)
    if (dx>=0) and (dy>=0):
        overlap = dx * dy
        iou = overlap / (a_area + b_area - overlap)
        return iou
    return 0

def transform_image(image, model_image_size):
    """ Transforms an image to be used as an input for a model
        Input:
            image (PIL.Image) - the image to be transformed
            model_image_size (numpy array) - the final size of the image
        Ouput:
            image_date (numpy array) - the transformed image as a numpy array to be used in a model 
     """

    resized_image = image.resize(model_image_size, Image.BICUBIC)
    image_data = np.array(resized_image, dtype=np.float32)

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data

def detect_from_model_output(out, anchors, num_classes):
    """ Detects objects based on model output. Dimensions of objects detected are independent of actual image size.
        Output must be scaled to image size.
        Input:
            out (numpy array) - the output of the detection model
            anchors (numpy array) - anchors used in the model
            num_classes (int) - the number of classes that were detected
        Output:
            box_xy (numpy array) - top left x,y coordinates in a tuple form
            box_wh (numpy array) - width and height of objects in tuple form
            box_confidence (numpy array) - confidence levels of respective objects detected
            box_class_probs (numpy array) - condiedence level of the respective boxes used to detect objects 
    """
    num_anchors = len(anchors)
    anchors = anchors.reshape([1,1,1,num_anchors,2])
    h, w = out.shape[1], out.shape[2]
    conv_dims = np.array([h,w]).reshape([1,1,1,1,2]) # 19 19
    conv_index = np.flip(np.array(list(product(range(h), repeat=2))),-1).reshape([1,h,w,1,2]).astype(out.dtype)
    out = out.reshape([-1, h, w, num_anchors, num_classes + 5])
    box_xy = sigmoid(out[..., :2])
    box_wh = np.exp(out[..., 2:4])
    box_confidence = sigmoid(out[..., 4:5])
    box_class_probs = softmax(out[..., 5:])
    
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors / conv_dims
    
    return box_xy, box_wh, box_confidence, box_class_probs

def fit_to_image(im, box_xy, box_wh, box_confidence, box_class_probs):
    """ Rescales formatted output to original image dimensions """
    im_xy = box_xy * np.array([im.width, im.height])
    im_xy = im_xy.reshape([-1,2])
    im_wh = box_wh * np.array([im.width, im.height])
    im_wh = im_wh.reshape([-1,2])

    im_xy -= (im_wh / 2.)

    im_confidence = box_confidence.flatten()
    im_class_probs = np.argmax(box_class_probs,axis=-1).flatten()
    
    return im_xy, im_wh, im_confidence, im_class_probs

def find_centers(im_xy, im_wh, im_confidence, im_class_probs, class_names, min_confidence, max_overlap):
    objects = {"time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    params = list(zip(im_xy, im_wh, im_confidence, im_class_probs))
    params.sort(key=lambda x: x[2])
    for (x,y), (w,h), confidence, class_name_idx in params:
        class_name = class_names[class_name_idx]
        if confidence > min_confidence:
            first = True
            coors = [(x,y), (x+w, y+h)]
            center = (x + (w / 2.), y + (h / 2.))
            xywh = (x,y,w,h)
            for other_obj in objects.get(class_name, []):
                other_coors = other_obj.get("coors")
                if iou(coors, other_coors) > max_overlap:
                    first = False
                    break
            if first:
                objects[class_name] = objects.get(class_name, []) + [{"coors": coors, "center": center, "x": x, "y": y, "w": w, "h": h, "xywh":xywh }]
    return objects

def detect_from_image(
        image, 
        anchors_path=ANCHORS_PATH,
        classes_path=CLASSES_PATH,
        config_path=CONFIG_PATH,
        weights_path=WEIGHTS_PATH,
        min_confidence=MIN_CONFIDENCE,
        max_overlap=MAX_OVERLAP,
        model_image_size=MODEL_IMAGE_SIZE
        ):

    if type(image) is str:
        image = Image.open(image)

    anchors = get_anchors(anchors_path)
    class_names, _ = get_classes(classes_path)
    num_classes = len(class_names)

    im_in = transform_image(image, model_image_size)

    global model

    if model is None:
        model = get_model(config_path, weights_path)
    
    out = model.predict(im_in)
    box_xy, box_wh, box_confidence, box_class_probs = detect_from_model_output(out,anchors,num_classes)
    im_xy, im_wh, im_confidence, im_class_probs = fit_to_image(image, box_xy, box_wh, box_confidence, box_class_probs)
    objects = find_centers(im_xy, im_wh, im_confidence, im_class_probs, class_names, min_confidence, max_overlap)
    
    return objects