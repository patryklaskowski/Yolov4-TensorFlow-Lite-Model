from TensorFlowLiteModel import create_prediction_stream
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-tf', '--tflite', dest = 'tflife', default = './checkpoints/yolov4-coco-416.tflite', help = 'Path to tfite model.')
parser.add_argument('-n', '--names', dest = 'names', default = './data/classes/coco.names', help = 'Path to .names file.')
args = parser.parse_args()

def main():
    model_path = './checkpoints/yolov4-coco-416.tflite'
    names_path = './data/classes/coco.names'
    video_source = 0 #'http://208.139.200.133/mjpg/video.mjpg' # 0
#     create_prediction_stream(model_path, names_path, video_source)
    create_prediction_stream(args.tflite, args.names, video_source)

if __name__ == '__main__':
    main()
