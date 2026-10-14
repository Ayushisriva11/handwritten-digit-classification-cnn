import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model


# --------------------------------------------------
# Load already-trained CNN model
# --------------------------------------------------

model = load_model("mnist_cnn_model.h5")


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("MNIST Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload an image of a handwritten digit",
    type=["png", "jpg", "jpeg"]
)


# --------------------------------------------------
# Process uploaded image
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        original_image,
        caption="Original Image",
        width=300
    )

    # Convert PIL → NumPy
    img = np.array(original_image)


    # --------------------------------------------------
    # Convert RGB → HSV
    # --------------------------------------------------

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    )


    # --------------------------------------------------
    # Detect blue ink
    # --------------------------------------------------

    lower_blue = np.array(
        [80, 40, 30]
    )

    upper_blue = np.array(
        [140, 255, 255]
    )

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )


    # --------------------------------------------------
    # Clean small noise
    # --------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # --------------------------------------------------
    # Find contours
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        blue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    digit_contour = None
    largest_area = 0


    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = cv2.contourArea(
            contour
        )

        if area < 5:
            continue

        # Ignore extremely thin objects
        aspect_ratio = w / float(h)

        if aspect_ratio < 0.15:
            continue

        if aspect_ratio > 5:
            continue

        if area > largest_area:

            largest_area = area
            digit_contour = contour


    # --------------------------------------------------
    # Check detection
    # --------------------------------------------------

    if digit_contour is None:

        st.error(
            "Could not detect the handwritten digit."
        )

    else:

        # --------------------------------------------------
        # Bounding box around digit
        # --------------------------------------------------

        x, y, w, h = cv2.boundingRect(
            digit_contour
        )

        margin = 10

        x1 = max(
            0,
            x - margin
        )

        y1 = max(
            0,
            y - margin
        )

        x2 = min(
            img.shape[1],
            x + w + margin
        )

        y2 = min(
            img.shape[0],
            y + h + margin
        )


        # --------------------------------------------------
        # Crop ONLY the blue mask
        # --------------------------------------------------

        digit_mask = blue_mask[
            y1:y2,
            x1:x2
        ]


        # --------------------------------------------------
        # Convert mask to PIL image
        # --------------------------------------------------

        digit_image = Image.fromarray(
            digit_mask
        )


        # --------------------------------------------------
        # Resize while maintaining aspect ratio
        # --------------------------------------------------

        digit_image.thumbnail(
            (20, 20)
        )


        # --------------------------------------------------
        # Create MNIST-style 28×28 image
        #
        # Background = black (0)
        # Digit = white (255)
        # --------------------------------------------------

        canvas = Image.new(
            "L",
            (28, 28),
            0
        )


        # Center digit
        paste_x = (
            28 - digit_image.width
        ) // 2

        paste_y = (
            28 - digit_image.height
        ) // 2


        canvas.paste(
            digit_image,
            (
                paste_x,
                paste_y
            )
        )


        # --------------------------------------------------
        # Convert to NumPy
        # --------------------------------------------------

        processed = np.array(
            canvas
        )


        # --------------------------------------------------
        # Normalize exactly to 0–1
        # --------------------------------------------------

        processed = (
            processed.astype(
                "float32"
            ) / 255.0
        )


        # --------------------------------------------------
        # CNN input shape
        # --------------------------------------------------

        processed = processed.reshape(
            1,
            28,
            28,
            1
        )


        # --------------------------------------------------
        # Display processed image
        # --------------------------------------------------

        st.subheader(
            "Processed 28×28 MNIST Image"
        )

        st.image(
            canvas,
            width=200
        )


        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = model.predict(
            processed,
            verbose=0
        )


        predicted_digit = int(
            np.argmax(prediction)
        )

        confidence = (
            float(
                np.max(prediction)
            ) * 100
        )


        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        st.subheader(
            f"Predicted Digit: {predicted_digit}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )