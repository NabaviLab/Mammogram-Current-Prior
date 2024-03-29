import os
import numpy as np
from tensorflow.keras.applications import ResNet50 # Changed from ResNet101 to ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
dataset_path = "./Classes/"

def load_images(image_directory, label):
    images = []
    labels = []
    for img_name in os.listdir(image_directory):
        img_path = os.path.join(image_directory, img_name)
        try:
            img = load_img(img_path, target_size=(1024, 1024))
            img = img_to_array(img)
            img = preprocess_input(img)
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
    return images, labels

def load_data(dataset_path):
    normal_images, normal_labels = load_images(os.path.join(dataset_path, 'Normal'), 0)
    abnormal_images, abnormal_labels = load_images(os.path.join(dataset_path, 'Abnormal'), 1)
    
    X = np.array(normal_images + abnormal_images)
    y = np.array(normal_labels + abnormal_labels)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def create_resnet_model(input_shape):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    return model

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True,
    fill_mode='nearest')

X_train, X_val, y_train, y_val = load_data(dataset_path)

model = create_resnet_model(input_shape=(1024, 1024, 3))

adam_optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=150,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint])

model.save('resnet50.h5')
