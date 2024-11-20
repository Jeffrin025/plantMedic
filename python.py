# # import numpy as np
# # import pickle
# # import cv2
# # from os import listdir, path
# # from sklearn.preprocessing import LabelBinarizer
# # import tensorflow as tf
# # from sklearn.model_selection import train_test_split
# # import matplotlib.pyplot as plt

# # # Configuration Constants
# # EPOCHS = 40
# # INIT_LR = 1e-3
# # BS = 32
# # default_image_size = (256, 256)
# # directory_root = r"C:\jaff\python\input\PlantVillage"  # Use raw string
# # width, height, depth = 256, 256, 3

# # def convert_image_to_array(image_dir):
# #     try:
# #         image = cv2.imread(image_dir)
# #         if image is not None:
# #             image = cv2.resize(image, default_image_size)
# #             return tf.keras.preprocessing.image.img_to_array(image)
# #         else:
# #             print(f"[ERROR] Image not found or could not be loaded: {image_dir}")
# #             return None
# #     except Exception as e:
# #         print(f"Error loading image {image_dir}: {e}")
# #         return None

# # image_list, label_list = [], []

# # try:
# #     print("[INFO] Loading images ...")
    
# #     # Check if the root directory exists
# #     if not path.exists(directory_root):
# #         print(f"[ERROR] Directory does not exist: {directory_root}")
# #         exit()

# #     # Get a list of plant folders in the root directory
# #     root_dir = [d for d in listdir(directory_root) if path.isdir(path.join(directory_root, d))]

# #     print(f"[INFO] Found root directories: {root_dir}")

# #     for plant_folder in root_dir:
# #         plant_folder_path = path.join(directory_root, plant_folder)
        
# #         # Check if the plant folder path exists
# #         if not path.exists(plant_folder_path):
# #             print(f"[ERROR] Plant folder does not exist: {plant_folder_path}")
# #             continue

# #         # Load images directly from the plant folder
# #         plant_images = [img for img in listdir(plant_folder_path) if img.endswith((".jpg", ".JPG", ".png", ".jpeg"))]
        
# #         print(f"[INFO] Processing folder: {plant_folder}, found {len(plant_images)} images.")

# #         for image in plant_images[:200]:  # Limit to 200 images
# #             image_directory = path.join(plant_folder_path, image)
# #             img_array = convert_image_to_array(image_directory)
# #             if img_array is not None:
# #                 image_list.append(img_array)
# #                 label_list.append(plant_folder)  # Use the folder name as the label

# #     print("[INFO] Image loading completed")  
# # except Exception as e:
# #     print(f"[ERROR] Error while loading images: {e}")

# # # Check the number of images and labels loaded
# # print(f"[INFO] Number of images loaded: {len(image_list)}")
# # print(f"[INFO] Number of labels loaded: {len(label_list)}")

# # # Check for labels before proceeding
# # if len(label_list) == 0:
# #     print("[ERROR] No labels found. Exiting...")
# #     exit()

# # # Label Binarization
# # label_binarizer = LabelBinarizer()
# # image_labels = label_binarizer.fit_transform(label_list)
# # pickle.dump(label_binarizer, open('label_transform.pkl', 'wb'))
# # n_classes = len(label_binarizer.classes_)
# # print(f"[INFO] Classes found: {label_binarizer.classes_}")

# # # Normalize images
# # np_image_list = np.array(image_list, dtype=np.float16) / 255.0

# # # Train-Test Split
# # print("[INFO] Splitting data into train and test sets")
# # x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state=42)

# # # Data Augmentation
# # aug = tf.keras.preprocessing.image.ImageDataGenerator(
# #     rotation_range=25, width_shift_range=0.1,
# #     height_shift_range=0.1, shear_range=0.2, 
# #     zoom_range=0.2, horizontal_flip=True, 
# #     fill_mode="nearest"
# # )

# # # Model Architecture
# # model = tf.keras.models.Sequential()
# # inputShape = (height, width, depth)
# # chanDim = -1

# # if tf.keras.backend.image_data_format() == "channels_first":
# #     inputShape = (depth, height, width)
# #     chanDim = 1

# # # Convolutional layers
# # model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
# # model.add(tf.keras.layers.Activation("relu"))
# # model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
# # model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
# # model.add(tf.keras.layers.Dropout(0.25))

# # model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
# # model.add(tf.keras.layers.Activation("relu"))
# # model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
# # model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
# # model.add(tf.keras.layers.Activation("relu"))
# # model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
# # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# # model.add(tf.keras.layers.Dropout(0.25))

# # model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same"))
# # model.add(tf.keras.layers.Activation("relu"))
# # model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
# # model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same"))
# # model.add(tf.keras.layers.Activation("relu"))
# # model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
# # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# # model.add(tf.keras.layers.Dropout(0.25))

# # # Fully connected layers
# # model.add(tf.keras.layers.Flatten())
# # model.add(tf.keras.layers.Dense(1024))
# # model.add(tf.keras.layers.Activation("relu"))
# # model.add(tf.keras.layers.BatchNormalization())
# # model.add(tf.keras.layers.Dropout(0.5))
# # model.add(tf.keras.layers.Dense(n_classes))
# # model.add(tf.keras.layers.Activation("softmax"))

# # # Compile Model
# # opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
# # model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# # # Train the model
# # print("[INFO] Training network...")
# # history = model.fit(
# #     aug.flow(x_train, y_train, batch_size=BS),
# #     validation_data=(x_test, y_test),
# #     steps_per_epoch=len(x_train) // BS,
# #     epochs=EPOCHS, verbose=1
# # )

# # # Plot training history
# # acc = history.history['accuracy']
# # val_acc = history.history['val_accuracy']
# # loss = history.history['loss']
# # val_loss = history.history['val_loss']
# # epochs = range(1, len(acc) + 1)

# # plt.plot(epochs, acc, 'b', label='Training accuracy')
# # plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
# # plt.title('Training and Validation Accuracy')
# # plt.legend()
# # plt.figure()

# # plt.plot(epochs, loss, 'b', label='Training loss')
# # plt.plot(epochs, val_loss, 'r', label='Validation loss')
# # plt.title('Training and Validation Loss')
# # plt.legend()
# # plt.show()


# # # Save the model
# # print("[INFO] Saving model...")
# # model.save('cnn_model.h5')
# # print("[INFO] Model saved as cnn_model.h5")


# import numpy as np
# import pickle
# import cv2
# from os import listdir, path
# from sklearn.preprocessing import LabelBinarizer
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# # Configuration Constants
# EPOCHS = 60
# INIT_LR = 1e-3  # Decreased learning rate
# BS = 32
# default_image_size = (256, 256)
# directory_root = r"C:\Users\91638\Downloads\Plant_leaf_diseases_dataset_with_augmentation\Plant_leave_diseases_dataset_with_augmentation"  # Use raw string
# width, height, depth = 256, 256, 3

# def convert_image_to_array(image_dir):
#     try:
#         image = cv2.imread(image_dir)
#         if image is not None:
#             image = cv2.resize(image, default_image_size)
#             return tf.keras.preprocessing.image.img_to_array(image)
#         else:
#             print(f"[ERROR] Image not found or could not be loaded: {image_dir}")
#             return None
#     except Exception as e:
#         print(f"Error loading image {image_dir}: {e}")
#         return None

# image_list, label_list = [], []

# try:
#     print("[INFO] Loading images ...")
    
#     # Check if the root directory exists
#     if not path.exists(directory_root):
#         print(f"[ERROR] Directory does not exist: {directory_root}")
#         exit()

#     # Get a list of plant folders in the root directory
#     root_dir = [d for d in listdir(directory_root) if path.isdir(path.join(directory_root, d))]

#     print(f"[INFO] Found root directories: {root_dir}")

#     for plant_folder in root_dir:
#         plant_folder_path = path.join(directory_root, plant_folder)
        
#         # Check if the plant folder path exists
#         if not path.exists(plant_folder_path):
#             print(f"[ERROR] Plant folder does not exist: {plant_folder_path}")
#             continue

#         # Load images directly from the plant folder
#         plant_images = [img for img in listdir(plant_folder_path) if img.endswith((".jpg", ".JPG", ".png", ".jpeg"))]
        
#         print(f"[INFO] Processing folder: {plant_folder}, found {len(plant_images)} images.")

#         for image in plant_images[:200]:  # Limit to 200 images
#             image_directory = path.join(plant_folder_path, image)
#             img_array = convert_image_to_array(image_directory)
#             if img_array is not None:
#                 image_list.append(img_array)
#                 label_list.append(plant_folder)  # Use the folder name as the label

#     print("[INFO] Image loading completed")  
# except Exception as e:
#     print(f"[ERROR] Error while loading images: {e}")

# # Check the number of images and labels loaded
# print(f"[INFO] Number of images loaded: {len(image_list)}")
# print(f"[INFO] Number of labels loaded: {len(label_list)}")

# # Check for labels before proceeding
# if len(label_list) == 0:
#     print("[ERROR] No labels found. Exiting...")
#     exit()

# # Label Binarization
# label_binarizer = LabelBinarizer()
# image_labels = label_binarizer.fit_transform(label_list)
# pickle.dump(label_binarizer, open('label_transform.pkl', 'wb'))
# n_classes = len(label_binarizer.classes_)
# print(f"[INFO] Classes found: {label_binarizer.classes_}")

# # Normalize images
# np_image_list = np.array(image_list, dtype=np.float16) / 255.0

# # Train-Test Split
# print("[INFO] Splitting data into train and test sets")
# x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state=42)

# # Increased Data Augmentation
# aug = tf.keras.preprocessing.image.ImageDataGenerator(
#     rotation_range=40,  # Increased rotation range
#     width_shift_range=0.2,  # Increased width shift range
#     height_shift_range=0.2,  # Increased height shift range
#     shear_range=0.2,
#     zoom_range=0.3,  # Increased zoom range
#     horizontal_flip=True,
#     fill_mode="nearest"
# )

# # Model Architecture with More Layers
# model = tf.keras.models.Sequential()
# inputShape = (height, width, depth)
# chanDim = -1

# if tf.keras.backend.image_data_format() == "channels_first":
#     inputShape = (depth, height, width)
#     chanDim = 1

# # Convolutional layers
# model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.BatchNormalization(axis=chanDim))  # Added Batch Normalization
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
# model.add(tf.keras.layers.Dropout(0.2))  # Adjusted Dropout Rate

# model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Dropout(0.2))  # Adjusted Dropout Rate

# model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same"))
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
# model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same"))
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Dropout(0.3))  # Adjusted Dropout Rate

# # Fully connected layers
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(1024))
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.5))  # Keep this dropout rate higher
# model.add(tf.keras.layers.Dense(n_classes))
# model.add(tf.keras.layers.Activation("softmax"))

# # Compile Model with Updated Learning Rate
# opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# # Train the model
# print("[INFO] Training network...")
# history = model.fit(
#     aug.flow(x_train, y_train, batch_size=BS),
#     validation_data=(x_test, y_test),
#     steps_per_epoch=len(x_train) // BS,
#     epochs=EPOCHS, verbose=1
# )

# # Plot training history
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'b', label='Training accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.figure()

# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

# # Save the model
# print("[INFO] Saving model...")
# model.save('cnn_model_1.h5')
# print("[INFO] Model saved as cnn_model_1.h5")




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.resnet import ResNet101
from tensorflow.keras.applications import ResNet50
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import EfficientNetV2L

import random
import os
import warnings
warnings.filterwarnings('ignore')
print('Done')
image_shape = (224,224)
batch_size = 64

train_dir ="C:/Users/91638/Downloads/archive (2)/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
valid_dir = "C:/Users/91638/Downloads/archive (2)/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

# apply scaling only becouse data already augmented
train_datagen = ImageDataGenerator(rescale=1/255., validation_split=0.2)
test_datagen = ImageDataGenerator(rescale = 1/255.)

# load training data
print("Training Images:")
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=image_shape,
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=True,
                                               subset='training')

# load validation data (20% of training data)
print("Validating Images:")
valid_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=image_shape,
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=False,
                                               subset='validation')

# load test data (consider validation data as test data)
print('Test Images:')
test_data = test_datagen.flow_from_directory(valid_dir,
                                               target_size=image_shape,
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=False)
# show how data store 
images, labels = next(iter(train_data))
print(f'shape of image is : {images[0].shape}')
print(f'label  \n{labels[0]}')
# show all diseases in dataset
diseases = os.listdir(train_dir)
print(diseases)
# identify uniqe plant in dataset
plants = []
NumberOfDiseases = 0
for plant in diseases:
    if plant.split('___')[0] not in plants:
        plants.append(plant.split('___')[0])
print(f'number of different plants is :{len(plants)}')
print(plants)
def plot_learning_curves(history):
    plt.figure(figsize=(12, 4))

    # accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # loss curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
def predict_labels_and_display(model_path, test_dir='/C:/Users/91638/Downloads/archive (2)/archive/test/test', image_size=(224, 224)):
    # load the best model
    best_model = load_model(model_path)

    true_labels = []
    predicted_labels = []
    images = []

    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')): 
            # load test images
            img_path = os.path.join(test_dir, filename)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # predict
            prediction = best_model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # extract the label (name of image)
            true_label = filename.split('.')[0]

            # get the prediction class
            class_labels = list(train_data.class_indices.keys())
            predicted_label = class_labels[predicted_class]

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            images.append(img)

    # randomly select three images
    selected_indices = random.sample(range(len(images)), 3)

    # show selected images
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(selected_indices):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[idx])
        plt.title(f'True: {true_labels[idx]}\nPredicted: {predicted_labels[idx]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    # Model Architecture
model = Sequential()

model.add(Conv2D(32,(3,3),activation = 'elu',input_shape=(224,224,3), kernel_initializer=GlorotNormal()))
model.add(Conv2D(32, (3,3), activation='elu', kernel_initializer=GlorotNormal()))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='elu', kernel_initializer=GlorotNormal()))
model.add(Conv2D(64, (3,3), activation='elu', kernel_initializer=GlorotNormal()))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='elu', kernel_initializer=GlorotNormal()))
model.add(Conv2D(128, (3,3), activation='elu', kernel_initializer=GlorotNormal()))
model.add(MaxPooling2D(2,2))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(256, activation='elu', kernel_initializer=GlorotNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(128, activation='elu', kernel_initializer=GlorotNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(64, activation='elu', kernel_initializer=GlorotNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(38, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# Suppress all TensorFlow logs except errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '0' = all messages are logged, '1' = filter out INFO, '2' = filter out WARNING, '3' = only errors are logged

# Optional: Disable XLA logs if you don’t need XLA
tf.config.optimizer.set_jit(False)  # Disable XLA compilation globally (if not needed)

# train the model
model_checkpoint = ModelCheckpoint('C:/jaff/models/cnn_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max', restore_best_weights=True)

history = model.fit(train_data,
                    validation_data=valid_data,
                    epochs=10,
                    batch_size=64, 
                    callbacks=[model_checkpoint, early_stopping])

