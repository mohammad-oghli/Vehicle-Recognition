import cv2
import numpy as np
from openvino.runtime import Core
from typing import Tuple


def model_init(model_path: str) -> Tuple:
    """
    Read the network and weights from file, load the
    model on the CPU and get input and output names of nodes

    :param: model: model architecture path *.xml
    :retuns:
            input_key: Input node network
            output_key: Output node network
            exec_net: Encoder model network
            net: Model network
    """
    # Initialize OpenVINO Runtime runtime.
    ie_core = Core()
    # Read the network and corresponding weights from a file.
    model = ie_core.read_model(model=model_path)
    # Compile the model for CPU (you can use GPU or MYRIAD as well).
    compiled_model = ie_core.compile_model(model=model, device_name="CPU")
    # Get input and output names of nodes.
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model


def crop_images(bgr_image, resized_image, boxes, threshold=0.6) -> np.ndarray:
    """
    Use bounding boxes from detection model to find the absolute car position

    :param: bgr_image: raw image
    :param: resized_image: resized image
    :param: boxes: detection model returns rectangle position
    :param: threshold: confidence threshold
    :returns: car_position: car's absolute position
    """
    # Fetch image shapes to calculate ratio
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Find the boxes ratio
    boxes = boxes[:, 2:]
    # Store the vehicle's position
    car_position = []
    # Iterate through non-zero boxes
    for box in boxes:
        # Pick confidence factor from last place in array
        conf = box[0]
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio
            # In case that bounding box is found at the top of the image,
            # we position upper box bar little bit lower to make it visible on image
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y * resized_y, 10)) if idx % 2
                else int(corner_position * ratio_x * resized_x)
                for idx, corner_position in enumerate(box[1:])
            ]

            car_position.append([x_min, y_min, x_max, y_max])

    return car_position


def vehicle_recognition(compiled_model_re, input_size, raw_image) -> Tuple:
    """
    Vehicle attributes recognition, input a single vehicle, return attributes
    :param: compiled_model_re: recognition net
    :param: input_size: recognition input size
    :param: raw_image: single vehicle image
    :returns: attr_color: predicted color
                       attr_type: predicted type
    """
    # An attribute of a vehicle.
    colors = ['White', 'Gray', 'Yellow', 'Red', 'Green', 'Blue', 'Black']
    types = ['Car', 'Bus', 'Truck', 'Van']

    # Resize the image to input size.
    resized_image_re = cv2.resize(raw_image, input_size)
    input_image_re = np.expand_dims(resized_image_re.transpose(2, 0, 1), 0)

    # Run inference.
    # Predict result.
    predict_colors = compiled_model_re([input_image_re])[compiled_model_re.output(1)]
    # Delete the dim of 2, 3.
    predict_colors = np.squeeze(predict_colors, (2, 3))
    predict_types = compiled_model_re([input_image_re])[compiled_model_re.output(0)]
    predict_types = np.squeeze(predict_types, (2, 3))

    attr_color, attr_type = (colors[np.argmax(predict_colors)],
                             types[np.argmax(predict_types)])
    return attr_color, attr_type
