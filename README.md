# Chinese Cafeteria Food Recognition

One of the common running themes in campus cafeteria is the hold up in foot traffic in queueing due to food checkout. Therefore we were inspired to build up an object detection model that recognizes entrees in the plate to perform automatic price calculations.

<img src="https://github.com/andywu0913/Chinese-Cafeteria-Food-Recognition/blob/experiencor/queue.jpg" width="550px">

A screenshot of the output predicted by the well-trained model.

<img src="https://github.com/andywu0913/Chinese-Cafeteria-Food-Recognition/blob/experiencor/model_demo.jpg" width="550px">

## Tested Compatible Environment
- Python 3.7
- NumPy 1.16.0
- Tensorflow 1.13.2
- Keras 2.2.5

## Usage

### Further Training
Train the model base on the settings in the `config.json`.
```
python train.py -c config.json
```

### Evaluation
One can evaluate the model accuracy with the following command after training.
```
python evaluate.py -c config.json
```

### Prediction

#### Single Image
```
python predict.py -c config.json -i /path/to/an/image [-o /path/to/output/folder/]
```

#### A Folder of Images
```
python predict.py -c config.json -i /path/to/image/folder [-o /path/to/output/folder/]
```

#### Live Webcam
```
python predict.py -c config.json -i webcam0
```

The number `0` in the parameter `webcam0` can be changed to the number of the order of the webcam installed on your device.

## Credits
Special thanks to experiencor/keras-yolo3 repository which provides an implementation of YOLOv3 by using Python, TensorFlow and Keras:)