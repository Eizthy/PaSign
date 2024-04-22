import cv2
import numpy as np
from keras.models import Model
from skimage.metrics import structural_similarity as ssim
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from django.db.models.fields.files import FieldFile
from django.core.files import File
from django.core.files.uploadedfile import InMemoryUploadedFile
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(300, 150, 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

def extract_features(img):
    # Resize image to match the model's expected sizing
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (150, 300))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    features = features.flatten()
    return features

# svm model is already trained and defined globally
def predict(img_file, signature_file):
    # Process img_file
    if isinstance(img_file, InMemoryUploadedFile):
        img_array = np.frombuffer(img_file.open().read(), np.uint8)
        if img_array.size == 0:
            raise ValueError("Empty image data")
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    elif isinstance(img_file, File):  # Assuming File is imported from django.core.files
        img = cv2.imread(img_file.path)
    else:
        raise ValueError("Unsupported image file type")

    # Process signature_file
    if isinstance(signature_file, FieldFile):
        signature_array = np.frombuffer(signature_file.read(), np.uint8)
        if signature_array.size == 0:
            raise ValueError("Empty signature data")
        signature = cv2.imdecode(signature_array, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Unsupported signature file type")
    
    img_features = extract_features(img)
    signature_features = extract_features(signature)

    
    similarity = (ssim(img_features, signature_features, data_range=img_features.max() - img_features.min()) * 100)

    return similarity

