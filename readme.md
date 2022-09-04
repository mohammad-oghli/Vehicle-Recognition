# Vehicle Detection & Recognition (Daisi Hackathon)

![Vehicle_Rec_Figure](https://i.imgur.com/dqPTlXl.jpg)

Python function as a web service to detect and recognize vehicles in image using machine learning.

The service uses two pre-trained models from Intel OpenVINO: [vehicle-detection-0200](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-detection-0200) for object detection and  [vehicle-attributes-recognition-barrier-0039](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-attributes-recognition-barrier-0039) for image classification.

The detection model is used to detect vehicle position, which is then cropped to a single vehicle before it is sent to a classification model to recognize attributes of the vehicle.

It will classify vehicles according to:
* **Type**: (Car, Bus, Truck, Van)
* **Color**: (White, Gray, Yellow, Red, Green, Blue, Black)

