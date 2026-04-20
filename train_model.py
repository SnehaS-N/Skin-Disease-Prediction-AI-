import os
import json
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "models/skin_model.h5"
CLASS_JSON_PATH = "class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

class_indices = train_data.class_indices
index_to_class = {str(v): k for k, v in class_indices.items()}

with open(CLASS_JSON_PATH, "w") as f:
    json.dump(index_to_class, f, indent=4)

print("Saved class mapping:", index_to_class)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)

print(f"Model saved at {MODEL_SAVE_PATH}")