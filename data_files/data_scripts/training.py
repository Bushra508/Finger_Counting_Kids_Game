import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# Dataset path
data_dir = "data_files\\data_sets\\data_grayscale_resized"

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    class_mode='sparse',
    subset='training',
    shuffle=True
)

# Validation data
val_data = val_datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

# Model architecture
model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_data.class_indices), activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping (optional)
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    # callbacks=[early_stop]
)

# Save model
model.save("dependencies\\model_30.keras")

# Evaluate model
y_true = val_data.classes
y_pred = np.argmax(model.predict(val_data), axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(6)]))
