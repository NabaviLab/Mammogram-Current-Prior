import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

model_path = 'h5 model loaded'
resnet_model = load_model(model_path)
base_model = Model(resnet_model.input, resnet_model.layers[-2].output)

def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def get_siamese_model(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    encoded_l = base_model(left_input)
    encoded_r = base_model(right_input)

    L2_layer = Lambda(lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=-1, keepdims=True)))
    L2_distance = L2_layer([encoded_l, encoded_r])

    # New layers for decision making
    classification_layer = Flatten()(L2_distance)
    classification_layer = Dense(128, activation='relu')(classification_layer)
    classification_layer = Dropout(0.5)(classification_layer)
    prediction = Dense(1, activation='sigmoid')(classification_layer)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_net

input_shape = (1024, 1024, 3)
siamese_model = get_siamese_model(input_shape)

def prepare_pairs(data_path, image_size=(1024, 1024)):
    prior_images = []
    current_images = []
    labels = []

    for patient_folder in sorted(os.listdir(data_path)):
        patient_path = os.path.join(data_path, patient_folder)
        prior_path = os.path.join(patient_path, 'Prior')
        current_path = os.path.join(patient_path, 'Current')

        prior_files = sorted(os.listdir(prior_path))
        current_files = sorted(os.listdir(current_path))

        for prior_img, current_img in zip(prior_files, current_files):
            img_path_prior = os.path.join(prior_path, prior_img)
            img_path_current = os.path.join(current_path, current_img)

            img_prior = load_img(img_path_prior, target_size=image_size)
            img_current = load_img(img_path_current, target_size=image_size)

            img_prior = img_to_array(img_prior)
            img_current = img_to_array(img_current)

            prior_images.append(img_prior)
            current_images.append(img_current)

            if 'MASS' or 'Calc' or 'Arch' in current_img:
                labels.append(1)
            else:
                labels.append(0)

    return np.array(prior_images), np.array(current_images), np.array(labels)

data_path = 'DatasetMammoHistory'
prior_images, current_images, labels = prepare_pairs(data_path)

prior_train, prior_test, current_train, current_test, labels_train, labels_test = train_test_split(
    prior_images, current_images, labels, test_size=0.2, random_state=42)

print(f"Loaded {len(prior_images)} prior images and {len(current_images)} current images.")

def visualize_pair(prior_image, current_image):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(prior_image.astype('uint8'))
    axs[0].title.set_text('Prior')
    axs[1].imshow(current_image.astype('uint8'))
    axs[1].title.set_text('Current')
    plt.show()

visualize_pair(prior_images[0], current_images[0])

siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
siamese_model.fit([prior_train, current_train], labels_train, batch_size=32, epochs=150)
