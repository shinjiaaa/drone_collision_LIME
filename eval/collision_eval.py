# 충돌 분류 모델 성능 평가 파일
import os
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

def load_dataset(base_dir, img_size=(128, 128)):
    all_images, all_labels = [], []

    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        images_dir = os.path.join(folder_path, "images")
        labels_path = os.path.join(folder_path, "labels.txt")

        if not os.path.isdir(folder_path) or not os.path.exists(labels_path):
            continue

        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        with open(labels_path, "r") as f:
            labels_list = []
            for line in f:
                line = line.strip()
                if line.isdigit():
                    val = int(line)
                    if val in [0, 1]:
                        labels_list.append(val)

        min_len = min(len(image_files), len(labels_list))
        image_files = image_files[:min_len]
        labels_list = labels_list[:min_len]

        for img_name, label in zip(image_files, labels_list):
            img_path = os.path.join(images_dir, img_name)
            img = load_img(img_path, target_size=img_size)
            img = img_to_array(img) / 255.0
            all_images.append(img)
            all_labels.append(label)

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    print(f"Loaded {len(all_images)} images from {base_dir}")
    return all_images, all_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (e.g. model_weights.h5)")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the validation/test dataset")
    args = parser.parse_args()

    X_test, y_test = load_dataset(args.test_dir)

    print(f"🔍 Loading model from: {args.model_path}")
    model = load_model(args.model_path)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n📉 Validation Loss: {loss:.4f}")
    print(f"✅ Validation Accuracy: {acc:.4f}\n")

    print("📊 Generating predictions...")
    y_pred = model.predict(X_test, verbose=1)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred_classes, digits=4, target_names=["No Collision (0)", "Collision (1)"]))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))