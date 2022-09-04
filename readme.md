# Vehicle Detection & Recognition (Daisi Hackathon)

![Vehicle_Rec_Figure](https://i.imgur.com/dqPTlXl.jpg)

Python function as a web service to detect and recognize vehicles in image using machine learning.

The service uses two pre-trained models from Intel OpenVINO: [vehicle-detection-0200](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-detection-0200) for object detection and  [vehicle-attributes-recognition-barrier-0039](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-attributes-recognition-barrier-0039) for image classification.

The detection model is used to detect vehicle position, which is then cropped to a single vehicle before it is sent to a classification model to recognize attributes of the vehicle.

It will classify vehicles according to:
* **Type**: (Car, Bus, Truck, Van)
* **Color**: (White, Gray, Yellow, Red, Green, Blue, Black)

### How to call it
* Load the Daisi
    <pre>
    import matplotlib.pyplot as plt
    import pydaisi as pyd
    vehicle_recognition = pyd.Daisi("oghli/Vehicle Recognition")</pre>
    
* Call the `vehicle_recognition` end point, passing the image source to process it, you can pass image source either from **images/** directory or from valid **url** of the image
    <pre>
    #image_source = "https://i.imgur.com/IvwQdz5.jpg"
    image_source = "images/car2.jpg"
    result = vehicle_recognition.cv_vehicle_detect(image_source).value
    result</pre>

* It will return **np array** representing:
processed image indicating detected vehicles info.

you can define this function to display output image:
<pre>
def plt_show(img):
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(img)
</pre>
Then display result:
<pre>
plt_show(result)
</pre>

Function `st_ui` included in the app to render the user interface of the application endpoints.

Also, you can use image samples in the **/images** directory to test it on the service.


