# Yolov4-TensorFlow-Lite-Model
Ready to run YOLOv4 implemented in TensorFlow Lite format.

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?style=plastic&logo=python&logoColor=yellow&labelColor=blue)](https://www.python.org/)
[![made-with-tensorflow](https://img.shields.io/badge/Made%20with-TensorFlow-1f425f.svg?style=plastic&logo=tensorflow&logoColor=orange&labelColor=blue)](https://www.tensorflow.org/)

---

## 1. Prepare environment
```
git clone https://github.com/patryklaskowski/Yolov4-TensorFlow-Lite-Model
cd Yolov4-TensorFlow-Lite-Model/
python3.7 -m venv env
source env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2. Download TensorFlow Lite Model
Go [here](https://github.com/patryklaskowski/Yolov4-TensorFlow-Lite-Model/tree/main/checkpoints) and download `yolov4-coco-416.tflite` into `./checkpoints/` directory.

## 3. Run
```
python run_example.py
```
