import sys

import dlib
from skimage import io
import cv2
import numpy as np

try:
    import face_recognition_models
except:
    print("Please install `face_recognition_models` with this command before using `face_recognition`:")
    print()
    print("pip install git+https://github.com/ageitgey/face_recognition_models")
    quit()


face_detector = dlib.get_frontal_face_detector()

predictor_model = face_recognition_models.pose_predictor_model_location()
pose_predictor = dlib.shape_predictor(predictor_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)



detector = dlib.get_frontal_face_detector()
options=dlib.get_frontal_face_detector()
options.num_threads = 4
options.be_verbose = True

hotdog = [-0.04732208,  0.0598151 ,  0.00324567, -0.05776324, -0.05849014,
       -0.03925902,  0.00490329, -0.10963184,  0.2171739 , -0.07274142,
        0.11410219, -0.01044829, -0.24215764,  0.02604566, -0.02498358,
        0.17927216, -0.2191972 , -0.07367396, -0.18240467, -0.13886872,
       -0.02637533,  0.11019464,  0.00842822,  0.05988787, -0.07974523,
       -0.28534371, -0.06229002, -0.09488696,  0.07656035, -0.02829991,
        0.02296838,  0.02538262, -0.17543381, -0.02473943,  0.07078177,
       -0.01786792, -0.03809625, -0.09742503,  0.22657064,  0.0885545 ,
       -0.1231515 , -0.01624694,  0.14053112,  0.23750323,  0.17415546,
        0.04628522,  0.02517225, -0.05248352,  0.11832859, -0.2669923 ,
        0.12500936,  0.10172579,  0.11491054,  0.10663124,  0.1649493 ,
       -0.20373803, -0.01827349,  0.09447934, -0.20444416,  0.12530786,
        0.10130947, -0.01292769, -0.06821815, -0.02381016,  0.18991387,
        0.14920582, -0.11593045, -0.10019872,  0.10818356, -0.20241012,
       -0.03776928,  0.00455614, -0.05292876, -0.12324437, -0.25841892,
       -0.00655669,  0.43756455,  0.1893017 , -0.17210209, -0.02920713,
       -0.04858894, -0.08333714,  0.11409805,  0.04802355, -0.07774992,
       -0.05571346, -0.01846742,  0.0393299 ,  0.17693974,  0.02745801,
       -0.0581798 ,  0.22458971,  0.0033083 , -0.08779698, -0.04447759,
        0.00764873, -0.13411714, -0.07110395, -0.07530349, -0.06996362,
        0.01627606, -0.03324171,  0.03246389,  0.04642355, -0.17912219,
        0.15984201,  0.0300494 , -0.07419328, -0.01234981,  0.09020486,
       -0.13588306, -0.00740595,  0.14930999, -0.20896034,  0.22638421,
        0.10970015, -0.0444756 ,  0.14580257, -0.04031482,  0.11743218,
        0.00449062,  0.0104951 , -0.03860776, -0.15778641, -0.00368786,
        0.02075585,  0.1185779 ,  0.04689222]
        

def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def load_image_file(filename, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array
    :param filename: image file to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    return scipy.misc.imread(filename, mode=mode)


def _raw_face_locations(img, number_of_times_to_upsample=1):
    """
    Returns an array of bounding boxes of human faces in a image
    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of dlib 'rect' objects of found face locations
    """
    return face_detector(img, number_of_times_to_upsample)


def face_locations(img, number_of_times_to_upsample=1):
    """
    Returns an array of bounding boxes of human faces in a image
    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample)]


def _raw_face_landmarks(face_image, face_locations=None):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_landmarks(face_image, face_locations=None):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image
    :param face_image: image to search
    :param face_locations: Optionally provide a list of face locations to check.
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """
    landmarks = _raw_face_landmarks(face_image, face_locations)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmarks_as_tuples]


def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.
    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimentional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)

    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.
    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

camera = cv2.VideoCapture(0)

list_face_locations = []
list_face_encodings = []
list_face_names = []

while True:
    ret, img = camera.read()    
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    #dets = detector(img, 1)

    list_face_locations = face_locations(img)
    list_face_encodings = face_encodings(img,list_face_locations)

    list_face_names = []

    for face_encoding in list_face_encodings:
        match = compare_faces([hotdog],face_encoding)
        name = "NOT hotdog"

        if match[0]:
            name = "hotdog"
        list_face_names.append(name)

        for (top,right,bottom,left), name in zip(list_face_locations,list_face_names):
            cv2.rectangle(img,(left,top),(right,bottom),(255,0,0),2)
            cv2.rectangle(img,(left,bottom-35),(right,bottom),cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img,name,(left+6,bottom-6),font,1.0,(255,255,255),1)

        cv2.imshow('detections',img)


        if cv2.waitKey(5) & 0x00 == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
