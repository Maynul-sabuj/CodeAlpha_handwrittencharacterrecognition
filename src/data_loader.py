
import string
import numpy as np
import tensorflow as tf


def _prep(X, y):
    X = X.astype("float32") / 255.0
    X = np.expand_dims(X, -1)  # add channel dim -> (N, 28, 28, 1)
    return X, y


def load_mnist():
    """Digits 0-9."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, y_train = _prep(X_train, y_train)
    X_test, y_test = _prep(X_test, y_test)
    class_names = [str(i) for i in range(10)]
    return (X_train, y_train), (X_test, y_test), class_names


def load_emnist(split: str = "letters"):
    """
    EMNIST characters, via the `emnist` package (pip install emnist).

    split="letters"  -> 26 classes, A-Z (uppercase), labels 1-26 remapped to 0-25
    split="balanced" -> 47 classes, digits + upper + a subset of lowercase
    """
    from emnist import extract_training_samples, extract_test_samples

    X_train, y_train = extract_training_samples(split)
    X_test, y_test = extract_test_samples(split)

    # EMNIST images are rotated/flipped relative to their canonical orientation
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    if split == "letters":
        # emnist 'letters' labels are 1-26 for A-Z; shift to 0-indexed
        y_train = y_train - 1
        y_test = y_test - 1
        class_names = list(string.ascii_uppercase)
    elif split == "balanced":
        # Standard EMNIST-balanced mapping (47 classes)
        chars = list(string.digits) + list(string.ascii_uppercase) + \
            ["a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"]
        class_names = chars
    else:
        raise ValueError(f"Unsupported EMNIST split: {split}")

    X_train, y_train = _prep(X_train, y_train)
    X_test, y_test = _prep(X_test, y_test)
    return (X_train, y_train), (X_test, y_test), class_names


def load_dataset(name: str):
    name = name.lower()
    if name == "mnist":
        return load_mnist()
    elif name in ("emnist", "emnist-letters"):
        return load_emnist("letters")
    elif name == "emnist-balanced":
        return load_emnist("balanced")
    else:
        raise ValueError(f"Unknown dataset '{name}'. Use: mnist, emnist-letters, emnist-balanced")
