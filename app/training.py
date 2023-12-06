import os
import cv2
import numpy as np
import tensorflow as tf
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from PIL import Image
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras import backend as K  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files import File
from django.db.models.fields.files import FieldFile
from skimage.metrics import structural_similarity as ssim
# from skimage.transform import resize

def get_random_image(img_size):
    random_image = np.random.rand(img_size[0], img_size[1], 3)
    return random_image

img_width, img_height = 300, 150
input_shape = (img_width, img_height, 1)
batch_size = 32


dataset_path = "C:\\Users\\ACER\\OneDrive\\Desktop\\Datasets\\Signature"
classes = os.listdir(dataset_path)
class_to_index = {class_name: i for i, class_name in enumerate(classes)}

images = []
labels = []

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        class_index = class_to_index[class_name]
        for forge_real in ['forge', 'real']:
            folder_path = os.path.join(class_path, forge_real)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (img_width, img_height))
                    img = img.reshape((img_width, img_height, 1)) 
                    images.append(img)
                    labels.append(class_index)

images = np.array(images)
labels = np.array(labels)


def create_siamese_model(input_shape=input_shape):
    model = Sequential()
    model.add(Convolution2D(16, (8, 8), strides=(1, 1), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(32, (4, 4), strides=(1, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, (2, 2), strides=(1, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='softmax'))  
    return model

siamese_model = create_siamese_model(input_shape)
rms = RMSprop()
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)
callback_early_stop_reduceLROnPlateau = [earlyStopping]

siamese_model.compile(optimizer=rms, loss='binary_crossentropy', metrics=["accuracy"])
siamese_model.summary()

one = []
zero = []

img_size = (300, 150)

for x in range(200):
    img = get_random_image(img_size)

    a, b = random.randrange(0, img_size[0] // 4), random.randrange(0, img_size[0] // 4)
    c, d = random.randrange(img_size[0] // 2, img_size[0]), random.randrange(img_size[0] // 2, img_size[0])

    value = random.sample([True, False], 1)[0]
    if value == False:
        img[a:c, b:d, 0] = 25
        img[a:c, b:d, 1] = 25
        img[a:c, b:d, 2] = 25
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).convert('L')) / 255
        one.append(img)
    else:
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).convert('L')) / 255
        zero.append(img)

additional_zero = np.array(zero).reshape(-1, img_width, img_height, 1)
additional_one = np.array(one).reshape(-1, img_width, img_height, 1)

images = np.concatenate([images, additional_one, additional_zero])
labels = np.concatenate([labels, np.ones(len(additional_one)), np.zeros(len(additional_zero))])

images = np.concatenate([images, additional_one, additional_zero])
labels = np.concatenate([labels, np.ones(len(additional_one)), np.zeros(len(additional_zero))])

shuffled_indices = np.random.permutation(len(images))
images = images[shuffled_indices]
labels = labels[shuffled_indices]

labels = labels.astype(int)  

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

total_sample_size = 50
test_sample_size = 200
dim1, dim2 = 300, 150

x_pair = np.zeros([total_sample_size, 2, dim1, dim2, 1])
y = np.zeros([total_sample_size, 1])

x_pair_test = np.zeros([test_sample_size, 2, dim1, dim2, 1])
y_test = np.zeros([test_sample_size, 1])

for x in range(total_sample_size):
    value = random.sample([True, False], 1)[0]
    if value:
        pair = random.choices(one, k=2)
        x_pair[x, 0, :, :, 0] = pair[0]
        x_pair[x, 1, :, :, 0] = pair[1]
        y[x] = 1
    else:
        x_pair[x, 0, :, :, 0] = random.choices(one, k=1)[0]
        x_pair[x, 1, :, :, 0] = random.choices(zero, k=1)[0]
        y[x] = 0

for x in range(test_sample_size):
    value = random.sample([True, False], 1)[0]
    if value:
        pair = random.choices(one, k=2)
        x_pair_test[x, 0, :, :, 0] = pair[0]
        x_pair_test[x, 1, :, :, 0] = pair[1]
        y_test[x] = 1
    else:
        x_pair_test[x, 0, :, :, 0] = random.choices(one, k=1)[0]
        x_pair_test[x, 1, :, :, 0] = random.choices(zero, k=1)[0]
        y_test[x] = 0

model2 = Model(inputs=siamese_model.input, outputs=siamese_model.layers[-2].output)

input_dim = (dim1, dim2, 1)

img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

feat_vecs_a = model2(img_a)
feat_vecs_b = model2(img_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
callback_early_stop_reduceLROnPlateau = [early_stopping]

model = Model(inputs=[img_a, img_b], outputs=distance)
model.compile(loss=contrastive_loss, optimizer=adam_optimizer, metrics=[accuracy])
model.summary()

model.fit([x_pair[:, 0], x_pair[:, 1]], y, validation_data=([x_pair_test[:, 0], x_pair_test[:, 1]], y_test), batch_size=batch_size, verbose=1, epochs=10, callbacks=callback_early_stop_reduceLROnPlateau)

model.save_weights('model_weights.h5')
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())
print('saved')

distances = model.predict([x_pair_test[:, 0], x_pair_test[:, 1]])

threshold = 0.5

binary_predictions = distances < threshold

svm_features = distances.flatten()

svm_threshold = 0.1
svm_labels = (svm_features < svm_threshold).astype(int)

svm_features = svm_features.reshape(-1, 1)

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    svm_features, svm_labels, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_svm, y_train_svm)

svm_predictions = svm_model.predict(X_test_svm)

svm_accuracy = accuracy_score(y_test_svm, svm_predictions)
print("SVM Accuracy: {:.2%}".format(svm_accuracy))
print("Classification Report:\n", classification_report(y_test_svm, svm_predictions))

true_positives = np.sum(np.logical_and(binary_predictions == 1, y_test == 1))
false_positives = np.sum(np.logical_and(binary_predictions == 1, y_test == 0))
true_negatives = np.sum(np.logical_and(binary_predictions == 0, y_test == 0))
false_negatives = np.sum(np.logical_and(binary_predictions == 0, y_test == 1))

FRR = false_negatives / (true_positives + false_negatives)
FAR = false_positives / (false_positives + true_negatives)
ACC = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)

print(f"Threshold: {threshold}")
print("False Rejection Rate (FRR): {:.2%}".format(FRR))
print("False Acceptance Rate (FAR): {:.2%}".format(FAR))
print("Accuracy Rate (ACC): {:.2%}".format(ACC))
print("="*50)

# Load Siamese model and create a feature extraction function
siamese_model = create_siamese_model(input_shape)  # Load your Siamese model
feature_extraction_model = Model(inputs=siamese_model.input, outputs=siamese_model.layers[-2].output)

def extract_features_siamese(img):
    img = cv2.resize(img, (img_width, img_height))
    img = img.reshape((1, img_width, img_height, 1))
    features = feature_extraction_model.predict(img)
    return features.flatten()

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
    
    img = img / 255.0 
    signature = signature / 255.0

    img_features_siamese = extract_features_siamese(img)
    signature_features_siamese = extract_features_siamese(signature)

    distance_siamese = np.linalg.norm(img_features_siamese - signature_features_siamese)

    similarity = ssim(img_features_siamese, signature_features_siamese, win_size=13, data_range=img_features_siamese.max() - img_features_siamese.min()) * 100
    svm_prediction_siamese = svm_model.predict([[distance_siamese]])

    print(f'{"="*50} svm_prediction_siamese got {"="*50}')
    print(f'distance_siamese: {distance_siamese}')
    print(f'similarity: {similarity}%')
    print(f'svm_prediction_siamese: {svm_prediction_siamese}')
    print(f'{"="*100}')

    return similarity