# Chinese Cafeteria Food Recognition

One of the common running themes in campus cafeteria is the hold up in foot traffic in queueing due to food checkout. Therefore we were inspired to build up an object detection model that recognizes entrees in the plate to perform automatic price calculations.

<img src="https://github.com/andywu0913/Chinese-Cafeteria-Food-Recognition/blob/experiencor/queue.jpg" width="550px">

## Tested Compatible Environment
- Python 3.7
- NumPy 1.16.0
- Tensorflow 1.13.2
- Keras 2.2.5

## Usage

### Training
```
python train.py -c config.json
```

### Evaluation
```
python evaluate.py -c config.json
```

### Prediction

#### Single Image
```
python predict.py -c config.json -i /path/to/an/image
```

#### A Folder of Images
```
python predict.py -c config.json -i /path/to/image/folder
```

#### Live Webcam
```
python predict.py -c config.json -i webcam0
```
```
python predict.py -c config.json -i webcam1
```
...