# src/model_utils.py
import os
import numpy as np
import mlflow
import mlflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Adjust this path to your dataset location or mount it in CI
ASL_TRAIN_DIR = os.environ.get("ASL_TRAIN_DIR", "data/asl_alphabet_train")

def get_train_val_gens(batch_size=BATCH_SIZE, img_size=IMG_SIZE, seed=SEED):
    train_datagen = ImageDataGenerator(
        preprocessing_function=None,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.25,
        shear_range=0.15,
        brightness_range=[0.7, 1.3],
        channel_shift_range=40.0,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        ASL_TRAIN_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=seed
    )
    val_gen = val_datagen.flow_from_directory(
        ASL_TRAIN_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=seed
    )
    return train_gen, val_gen

def load_model_from_registry(model_name, stage="Production"):
    """
    Load model from MLflow model registry.
    model_name e.g. "ASL-ResNet-Production"
    stage e.g. "Production" or "Staging"
    """
    model_uri = f"models:/{model_name}/{stage}"
    # mlflow.keras.load_model works if logged via mlflow.keras.log_model
    model = mlflow.keras.load_model(model_uri)
    return model

def save_confusion_matrix(cm, out_path="confusion_matrix.npy"):
    np.save(out_path, cm)
    return out_path

