# ==========================
# CNN Cheetah vs Tiger Classifier (from scratch)
# ==========================

import tensorflow as tf
from tensorflow.keras import layers, models
import os

BASE_DIR = "Tiger_Cheetah"
TRAIN_DIR = os.path.join(BASE_DIR,"train")
VAL_DIR   = os.path.join(BASE_DIR,"val")

IMG_SIZE = (400,400)
BATCH = 16

# ===== Load Data with Augmentation =====
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="binary"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="binary"
)

train_ds = train_ds.shuffle(512).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

data_augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.15),
])

# ===== Build CNN =====
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE,3)),
    data_augment,
    layers.Rescaling(1./255),

    layers.Conv2D(32,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(256,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256,activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1,activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===== Train =====
EPOCHS = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ===== Save Model =====
model.save("cheetah_tiger_cnn.h5")

# ===== Predict Example =====
# Usage:
# img = tf.keras.preprocessing.image.load_img("some.jpg", target_size=IMG_SIZE)
# x = tf.keras.preprocessing.image.img_to_array(img)
# x = tf.expand_dims(x,0)/255.0
# p = model.predict(x)[0][0]
# print("Tiger" if p>0.5 else "Cheetah", p)
