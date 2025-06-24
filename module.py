import os
import cv2
import math
import random
import numpy as np
import datetime as datasets
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd


class classifier_model():
# model architecture
    def __init__(self, dataset_dir, classes_list):
        # classes name
        self.CLASSES_LIST = classes_list
        # quantity of frames to be evaluated per video
        self.SEQUENCE_LENGTH = 20

        # dataset dir
        self.DATASET_DIR = dataset_dir

        # standard height and width
        self.IMAGE_HEIGHT = 64
        self.IMAGE_WIDTH = 64

        # for training replication purposes
        self.seed_constant = 19
        np.random.seed(self.seed_constant)
        random.seed(self.seed_constant)
        tf.random.set_seed(self.seed_constant)

        self.one_hot_encoded_labels = None

        # create model
        # self.architecture()


    def create_dataset(self):
        features = []
        labels = []
        video_files_path = []

        for class_index, class_name in enumerate(self.CLASSES_LIST):
            print(f"Extracting data of {class_name}")

            files_list = os.listdir(os.path.join(self.DATASET_DIR, class_name))

            for file_name in files_list:
                video_file_path = os.path.join(self.DATASET_DIR, class_name, file_name)

                frames = self.frame_features_extraction(video_file_path)

                if len(frames) == self.SEQUENCE_LENGTH:
                    features.append(frames)
                    labels.append(class_index)
                    video_files_path.append(video_file_path)

        # features and labels
        self.features = np.asarray(features)
        self.labels = np.asarray(labels)
        # self.video_files_path = video_files_path

        # one hot encoded labels
        self.one_hot_encoded_labels = to_categorical(self.labels)
        
        # spliting dataset for training and test
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(
                    self.features, self.one_hot_encoded_labels, test_size=0.15, shuffle = True,
                    random_state = self.seed_constant)

    def frame_features_extraction(self, video_path):
        frame_list = []

        video_reader = cv2.VideoCapture(video_path)
        video_frames_count =  int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count/self.SEQUENCE_LENGTH), 1)

        for frame_counter in range(self.SEQUENCE_LENGTH):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter*skip_frames_window)

            sucess, frame = video_reader.read()

            if not sucess:
                break

            resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            resized_frame = resized_frame/255
            frame_list.append(resized_frame)

        video_reader.release()

        return frame_list

    def architecture(self):
        self.model = Sequential()

        self.model.add(ConvLSTM2D(filters=4, kernel_size= (3,3), activation="tanh",
                            data_format="channels_last", recurrent_dropout=0.2,
                            return_sequences=True, input_shape=(self.SEQUENCE_LENGTH,
                                                    self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)
                            ))
        self.model.add(MaxPooling3D(pool_size=(1,2,2), padding="same", data_format="channels_last"))
        self.model.add(TimeDistributed(Dropout(0.2)))

        self.model.add(ConvLSTM2D(filters=14, kernel_size= (3,3), activation="tanh",
                            data_format="channels_last", recurrent_dropout=0.2,
                            return_sequences=True, input_shape=(self.SEQUENCE_LENGTH,
                                                    self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)
                            ))
        self.model.add(MaxPooling3D(pool_size=(1,2,2), padding="same", data_format="channels_last"))
        self.model.add(TimeDistributed(Dropout(0.2)))

        self.model.add(ConvLSTM2D(filters=16, kernel_size= (3,3), activation="tanh",
                            data_format="channels_last", recurrent_dropout=0.2,
                            return_sequences=True, input_shape=(self.SEQUENCE_LENGTH,
                                                    self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)
                            ))
        self.model.add(MaxPooling3D(pool_size=(1,2,2), padding="same", data_format="channels_last"))
        # model.add(TimeDistributed(Dropout(0.2)))
        self.model.add(Flatten())
        self.model.add(Dense(len(self.CLASSES_LIST), activation="softmax"))  

        # summary of the model
        self.model.summary()

    def predict(self, video_file_path, output_file_path="./output_video.mp4"):
        video_reader = cv2.VideoCapture(str(video_file_path))
        
        original_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                    video_reader.get(cv2.CAP_PROP_FPS), (original_width, original_height))

        frames_queue = deque(maxlen=self.SEQUENCE_LENGTH)

        predicted_class = ""

        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            normalized_frame = resized_frame/255.0

            frames_queue.append(normalized_frame)

            if len(frames_queue) == self.SEQUENCE_LENGTH:
                labels_probabilities = self.model.predict(np.expand_dims(frames_queue, axis=0))[0]
                predicted_label = np.argmax(labels_probabilities)
                print(labels_probabilities)
                predicted_class = self.CLASSES_LIST[predicted_label]

            cv2.putText(frame, predicted_class, (10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
            video_writer.write(frame)
        
        video_reader.release()
        video_writer.release()

        print(f"Prediction saved in {output_file_path}")
        
    def train(self, epochs, file="./model.weights.h5", model_path=None, validation_split=0.15, patience=10, best_weights=True, batch_size=4, resume=False, last_epoch=0):
        self.architecture()
        
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=patience, mode="min", restore_best_weights=True)
        self.model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy", Precision(name="precision"), Recall(name="recall")])

        self.model_history = self.model.fit(x=self.features_train, y=self.labels_train, epochs=epochs,
                                            batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping_callback])

    def load_model(self,path):
        self.model = tf.keras.models.load_model(path)
        print("Model loaded successfully :)")
        
    def save_architecture_image(self, path):
        try:
            path = os.path.join(path, "architecture.png")
            plot_model(self.model, to_file=path, show_shapes=True, show_layer_activations=True, show_layer_names=True)
        except Exception as e:
            print(f"Error: {e}")

    def save_model(self, path="./"):
        path = os.path.join(path, "model.keras")

        try:
            self.model.save(path)
            print(f"Model saved successfully in {path}")
        except Exception as e:
            print(f"Error: {e}")

    def save_metrics(self):
        try:
            print(self.model_history.history)
            metrics_df = pd.DataFrame(self.model_history.history)
            metrics_df.to_csv("metrics_history.csv", index_label="epoch")
            print("Métricas salvas com sucesso!")
        except Exception as e:
            print(f"Erro ao salvar métricas: {e}")
