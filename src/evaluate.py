# src/evaluate.py
import os
import numpy as np
import mlflow
from sklearn.metrics import classification_report, confusion_matrix
from model_utils import get_val_gen, load_local_h5, load_model_from_registry, save_confusion_matrix

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
MODEL_NAME = os.environ.get("MODEL_NAME", "ASL-ResNet-Production")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_H5_PATH", "model/resnet50_asl_augmented_frozen.h5")

def load_model_with_fallback():
    if MLFLOW_TRACKING_URI:
        try:
            print("Trying to load from MLflow registry...")
            return load_model_from_registry(MODEL_NAME, stage="Production", tracking_uri=MLFLOW_TRACKING_URI)
        except Exception as e:
            print("Registry load failed:", e)
            print("Falling back to local .h5:", LOCAL_MODEL_PATH)
    return load_local_h5(LOCAL_MODEL_PATH)

if __name__ == "__main__":
    model = load_model_with_fallback()
    val_gen = get_val_gen()
    steps = int((val_gen.samples + val_gen.batch_size - 1) / val_gen.batch_size)

    y_true_all = []
    y_pred_all = []
    val_gen.reset()
    for i in range(steps):
        Xb, yb = next(val_gen)
        preds = model.predict(Xb)
        y_pred_all.extend(preds.argmax(axis=1).tolist())
        y_true_all.extend(yb.argmax(axis=1).tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    overall_acc = (y_pred_all == y_true_all).mean()
    print("Validation accuracy:", overall_acc)

    target_names = [k for k, v in sorted(val_gen.class_indices.items(), key=lambda x: x[1])]
    print(classification_report(y_true_all, y_pred_all, target_names=target_names))
    cm = confusion_matrix(y_true_all, y_pred_all)
    print("Confusion matrix shape:", cm.shape)

    # Optional MLflow logging
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("monitoring_experiments")
        with mlflow.start_run():
            mlflow.log_param("model_source", LOCAL_MODEL_PATH)
            mlflow.log_metric("validation_accuracy", float(overall_acc))
            out = save_confusion_matrix(cm)
            mlflow.log_artifact(out)
            print("Logged results to MLflow.")

