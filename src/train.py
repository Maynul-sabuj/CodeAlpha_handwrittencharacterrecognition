

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

from data_loader import load_dataset


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(6, len(class_names) * 0.35), max(5, len(class_names) * 0.35)))
    sns.heatmap(cm, annot=len(class_names) <= 15, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist",
                         choices=["mnist", "emnist-letters", "emnist-balanced"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="../models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    (X_train, y_train), (X_test, y_test), class_names = load_dataset(args.dataset)
    num_classes = len(class_names)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Classes: {num_classes}")

    np.save(os.path.join(args.out_dir, "class_names.npy"), np.array(class_names))

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        callbacks.ModelCheckpoint(os.path.join(args.out_dir, "char_model.h5"),
                                   monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
    )

    plot_history(history, os.path.join(args.out_dir, "training_curves.png"))

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    plot_confusion_matrix(y_test, y_pred, class_names,
                           os.path.join(args.out_dir, "confusion_matrix.png"))

    print(f"\nModel and plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
