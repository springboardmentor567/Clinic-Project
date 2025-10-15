# yolo_vindr_cxr_resume.py
from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt

# ---------------------------
# 1. Paths
# ---------------------------
DATA_YAML = "D:/vinbigdata_yolo/data.yaml"  
TRAIN_MODEL_NAME = "VinDr_CXR_model2_resume"   # New folder for resumed training
CHECKPOINT_PATH = r"C:\Users\kesha\OneDrive\Desktop\info\runs\detect\VinDr_CXR_model2\weights\last.pt"  
TEST_IMAGE = "D:/vinbigdata_yolo/test/images/00000001.png"  
SAVE_PREDICTIONS = True

# ---------------------------
# 2. Resume Training YOLOv8
# ---------------------------
def resume_training():
    """
    Resumes YOLOv8 training from last checkpoint
    """
    print("✅ Resuming YOLOv8 training...")
    model = YOLO(CHECKPOINT_PATH)   # Resume from last.pt
    model.train(
        data=DATA_YAML,
        epochs=50,   # TOTAL epochs (if you trained 26, it will continue until 50)
        imgsz=640,
        batch=16,
        name=TRAIN_MODEL_NAME,
        resume=True
    )
    print("✅ Training resumed successfully!")
    return f"runs/detect/{TRAIN_MODEL_NAME}/weights/best.pt"

# ---------------------------
# 3. Predict and Visualize
# ---------------------------
def predict_image(model_path, image_path):
    print(f"✅ Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"✅ Predicting image: {image_path}")
    results = model.predict(source=image_path, conf=0.25)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if len(results[0].boxes) > 0:
        for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{int(cls_id)}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    if SAVE_PREDICTIONS:
        pred_dir = os.path.join("runs", "detect", "predict_resume")
        os.makedirs(pred_dir, exist_ok=True)
        save_path = os.path.join(pred_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"✅ Prediction saved at {save_path}")

# ---------------------------
# 4. Main Execution
# ---------------------------
if __name__ == "__main__":
    # Resume Training
    best_model_path = resume_training()

    # Predict after training
    predict_image(best_model_path, TEST_IMAGE)
