# src/model_utils.py
import os
import numpy as np
import mlflow
import mlflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model as keras_load_model

IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
SEED = int(os.environ.get("SEED", 42))

# IMPORTANT: your model path (as you confirmed)
LOCAL_H5_PATH = os.environ.get("LOCAL_H5_PATH", "model/resnet50_asl_augmented_frozen.h5")

# Dataset path (change or export ASL_TRAIN_DIR environment variable)
ASL_TRAIN_DIR = os.environ.get("ASL_TRAIN_DIR", "data/asl_alphabet_train")

def get_val_gen(batch_size=BATCH_SIZE, img_size=IMG_SIZE, seed=SEED, data_dir=ASL_TRAIN_DIR):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    val_datagen = ImageDataGenerator(validation_split=0.2)
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=seed
    )
    return val_gen

def load_local_h5(path=None):
    path = path or LOCAL_H5_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local model file not found at: {path}")
    return keras_load_model(path)

def load_model_from_registry(name, stage="Production", tracking_uri=None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{name}/{stage}"
    return mlflow.keras.load_model(model_uri)

def save_confusion_matrix(cm, out_path="confusion_matrix.npy"):
    np.save(out_path, cm)
    return out_path
