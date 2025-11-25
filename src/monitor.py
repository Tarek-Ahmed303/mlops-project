# src/monitor.py
import os
import numpy as np
import mlflow
from model_utils import get_val_gen, load_local_h5, load_model_from_registry

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
MODEL_NAME = os.environ.get("MODEL_NAME", "ASL-ResNet-Production")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_H5_PATH", "model/resnet50_asl_augmented_frozen.h5")
THRESHOLD = float(os.environ.get("MONITOR_THRESHOLD", 0.90))
MAX_BATCHES = int(os.environ.get("MONITOR_MAX_BATCHES", 3))

def load_model_with_fallback():
    if MLFLOW_TRACKING_URI:
        try:
            print("Trying to load model from MLflow registry...")
            return load_model_from_registry(MODEL_NAME, stage="Production", tracking_uri=MLFLOW_TRACKING_URI)
        except Exception as e:
            print("Registry load failed:", e)
            print("Falling back to local .h5:", LOCAL_MODEL_PATH)
    return load_local_h5(LOCAL_MODEL_PATH)

def monitor_and_alert(threshold=THRESHOLD, max_batches=MAX_BATCHES):
    model = load_model_with_fallback()
    val_gen = get_val_gen()
    X_list, y_list = [], []
    val_gen.reset()
    for i in range(max_batches):
        try:
            Xb, yb = next(val_gen)
        except StopIteration:
            break
        X_list.append(Xb)
        y_list.append(yb)

    if len(X_list) == 0:
        print("No batches collected for monitoring.")
        return

    X_eval = np.vstack(X_list)
    y_eval = np.vstack(y_list)
    preds = model.predict(X_eval)
    pred_labels = preds.argmax(axis=1)
    true_labels = y_eval.argmax(axis=1)
    acc = (pred_labels == true_labels).mean()
    print("Monitoring accuracy:", acc)

    # log monitor run
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("monitoring_experiments")
        with mlflow.start_run(run_name="monitor-run"):
            mlflow.log_metric("monitor_accuracy", float(acc))
            print("Logged monitor_accuracy to MLflow.")

    if acc < threshold:
        print(f"ALERT: accuracy {acc:.3f} < threshold {threshold}")
        # exit code non-zero will mark GH Action job as failed if you want that
        raise SystemExit(2)
    else:
        print("Model healthy - no drift detected.")
        return True

if __name__ == "__main__":
    monitor_and_alert()

