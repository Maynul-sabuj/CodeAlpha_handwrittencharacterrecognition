

import argparse
import numpy as np
from PIL import Image
import tensorflow as tf


def preprocess_image(path: str) -> np.ndarray:
    """
    Load an image, convert to grayscale, resize to 28x28, normalize.
    Handles both 'black char on white background' and 'white char on black
    background' inputs by auto-inverting when the image is mostly light
    (MNIST/EMNIST convention is white stroke on black background).
    """
    img = Image.open(path).convert("L").resize((28, 28))
    arr = np.array(img).astype("float32")

    if arr.mean() > 127:  # mostly light background -> invert to match training data
        arr = 255.0 - arr

    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # -> (1, 28, 28, 1)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="../models")
    args = parser.parse_args()

    model = tf.keras.models.load_model(f"{args.model_dir}/char_model.h5")
    class_names = np.load(f"{args.model_dir}/class_names.npy", allow_pickle=True)

    x = preprocess_image(args.image)
    probs = model.predict(x)[0]
    pred_idx = np.argmax(probs)

    print(f"Predicted character: {class_names[pred_idx]}")
    top5 = np.argsort(probs)[::-1][:5]
    print("Top-5 predictions:")
    for idx in top5:
        print(f"  {class_names[idx]:>3s}: {probs[idx]:.3f}")


if __name__ == "__main__":
    main()
