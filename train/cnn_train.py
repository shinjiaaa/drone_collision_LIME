# 충돌 분류 모델 train 파일
import os
import argparse
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

def load_dataset(base_dir, img_size=(128, 128)):
    all_images, all_labels = [], []

    for folder_name in os.listdir(base_dir):
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
                if line.isdigit():  # 숫자인 경우만
                    val = int(line)
                    if val in [0, 1]:
                        labels_list.append(val)

        # 개수 다르면 짧은 쪽 기준
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

def build_cnn_model(input_shape=(128, 128, 3)):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_rootdir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.experiment_rootdir, exist_ok=True)

    X_train, y_train = load_dataset(args.train_dir)
    X_val, y_val = load_dataset(args.val_dir)

    # 클래스 가중치 계산
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_train),
                                         y=y_train)
    class_weights = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights}")

    model = build_cnn_model()

    checkpoint_path = os.path.join(args.experiment_rootdir, "model_weights.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    print(f"Training complete! Best model saved at: {checkpoint_path}")
