import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from openvino.runtime import Core
from model_engine import model_init, crop_images, vehicle_recognition
from helper import load_image, to_rgb, plt_show

# Global Model Configuration
# A directory where the model will be downloaded.
base_model_dir = "model"
# The name of the model from Open Model Zoo.
detection_model_name = "vehicle-detection-0200"
recognition_model_name = "vehicle-attributes-recognition-barrier-0039"
# Selected precision (FP32, FP16, FP16-INT8)
precision = "FP32"

# Check if the model exists.
detection_model_path = (
    f"model/intel/{detection_model_name}/{precision}/{detection_model_name}.xml"
)
recognition_model_path = (
    f"model/intel/{recognition_model_name}/{precision}/{recognition_model_name}.xml"
)
# de -> detection
# re -> recognition
# Detection model initialization.
input_key_de, output_keys_de, compiled_model_de = model_init(detection_model_path)
# Recognition model initialization.
input_key_re, output_keys_re, compiled_model_re = model_init(recognition_model_path)

# Get input size - Detection.
height_de, width_de = list(input_key_de.shape)[2:]
# Get input size - Recognition.
height_re, width_re = list(input_key_re.shape)[2:]


def cv_vehicle_detect(image_source) -> np.ndarray:
    # Define colors for boxes and descriptions.
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}
    # Convert the base image from BGR to RGB format.
    image_de = load_image(image_source)
    if image_de is not None:
        rgb_image = to_rgb(image_de)
        resized_image_de = cv2.resize(image_de, (width_de, height_de))
        # Expand the batch channel to [1, 3, 256, 256].
        input_image_de = np.expand_dims(resized_image_de.transpose(2, 0, 1), 0)
        # Run inference.
        boxes = compiled_model_de([input_image_de])[output_keys_de]
        # Delete the dim of 0, 1.
        boxes = np.squeeze(boxes, (0, 1))
        # Remove zero only boxes.
        boxes = boxes[~np.all(boxes == 0, axis=1)]
        # Find positions of cars.
        car_position = crop_images(image_de, resized_image_de, boxes)
        for x_min, y_min, x_max, y_max in car_position:
            # Run vehicle recognition inference.
            attr_color, attr_type = vehicle_recognition(compiled_model_re, (72, 72),
                                                        image_de[y_min:y_max, x_min:x_max])
            # Close the window with a vehicle.
            plt.close()
            # Draw a bounding box based on position.
            # Parameters in the `rectangle` function are: image, start_point, end_point, color, thickness.
            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 6)
            # Print the attributes of a vehicle.
            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_min + 250, y_min - 60), colors["green"], -1)
            # Parameters in the `putText` function are: img, text, org, fontFace, fontScale, color, thickness, lineType.
            rgb_image = cv2.putText(
                rgb_image,
                f"{attr_color} {attr_type}",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                colors["red"],
                5,
                cv2.LINE_AA
            )

        return rgb_image

    return "Sorry, loading image failed"


def st_ui():
    '''
    Render the User Interface of the application endpoints
    '''
    st.title("Vehicle Detection & Recognition")
    st.caption("Image Recognition")
    st.info("Intel OpenVINO Model Implementation by Oghli")
    # hint = st.empty()
    # with hint.container():
    #     st.write("### The model detects vehicles and classify it according to:")
    #     st.write("####  • Type")
    #     st.write("####  • Color")
    st.sidebar.subheader("Upload image to detect vehicles")
    uploaded_image = st.sidebar.file_uploader("Upload image", type=["png", "jpg"],
                                              accept_multiple_files=False, key=None,
                                              help="Image to detect vehicles")
    s_msg = st.empty()
    example_image = load_image('images/test.jpg')
    st.subheader("Input Image")
    example = st.image(to_rgb(example_image))
    if example:
        placeholder = st.empty()
        de_btn = placeholder.button('Detect', key='1')
    if uploaded_image:
        placeholder.empty()
        example.empty()
        # hint.empty()
        st.image(uploaded_image)
        de_btn = st.button("Detect")
        s_msg = st.sidebar.success("Image uploaded successfully")

    if de_btn:
        s_msg.empty()
        # hint.empty()
        image_de = 'images/test.jpg'
        if uploaded_image:
            image_de = uploaded_image
        with st.spinner('Processing Image ...'):
            detection_image = cv_vehicle_detect(image_de)
            if not example:
                example.empty()
                st.image(image_de)
            st.subheader("Result Image")
            st.image(detection_image)
            # byte_detect_img = pil_to_bytes(detection_image)
            # st.download_button(label="Download Result", data=byte_super_img,
            #                    file_name="detect_image.jpeg", mime="image/jpeg")


if __name__ == "__main__":
    # render the app using streamlit ui function
    st_ui()
    # image_source = "images/test.jpg"
    # detect_image = cv_vehicle_detect(image_source)
    # plt_show(detect_image)
