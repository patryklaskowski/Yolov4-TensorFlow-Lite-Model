from TensorFlowLiteModel import create_prediction_stream
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-tf', '--tflite', dest = 'tflite', default = './checkpoints/yolov4-coco-416.tflite', help = 'Path to tfite model.')
parser.add_argument('-n', '--names', dest = 'names', default = './data/classes/coco.names', help = 'Path to .names file.')
parser.add_argument('-s', '--size', dest = 'size', default = 416, help = 'Input size', type=int)
args = parser.parse_args()

def main():
    video_source = 0 #'http://208.139.200.133/mjpg/video.mjpg' # 0
    create_prediction_stream(args.tflite, args.names, video_source, input_size=args.size)

if __name__ == '__main__':
    main()
