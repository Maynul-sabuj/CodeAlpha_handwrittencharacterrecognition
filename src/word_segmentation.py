

import argparse
import cv2
import numpy as np
import tensorflow as tf


def segment_characters(image_path: str):
    """Return a list of (x, cropped_28x28_char_image) sorted left-to-right."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Ensure white strokes on black background (EMNIST/MNIST convention)
    if img.mean() > 127:
        img = 255 - img

    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 20]
    boxes.sort(key=lambda b: b[0])  # left-to-right reading order

    chars = []
    for (x, y, w, h) in boxes:
        crop = thresh[y:y + h, x:x + w]
        # pad to square then resize to 28x28, keeping the character centered
        size = max(w, h) + 10
        square = np.zeros((size, size), dtype=np.uint8)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        square[y_off:y_off + h, x_off:x_off + w] = crop
        resized = cv2.resize(square, (28, 28))
        chars.append((x, resized))

    return chars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="../models")
    args = parser.parse_args()

    model = tf.keras.models.load_model(f"{args.model_dir}/char_model.h5")
    class_names = np.load(f"{args.model_dir}/class_names.npy", allow_pickle=True)

    chars = segment_characters(args.image)
    if not chars:
        print("No characters detected. Try a higher-contrast image.")
        return

    predicted = []
    for _, char_img in chars:
        x = char_img.astype("float32") / 255.0
        x = np.expand_dims(x, axis=(0, -1))
        probs = model.predict(x, verbose=0)[0]
        predicted.append(str(class_names[np.argmax(probs)]))

    print(f"Detected {len(chars)} characters.")
    print("Predicted word:", "".join(predicted))


if __name__ == "__main__":
    main()
