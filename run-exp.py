import librosa
import sys
import argparse
import tensorflow as tf
from tensorflow import keras

def load_data(dir):
    pass

def main(args):
    train_dir = args.train_dir
    test_dir = args.test_dir
    print(keras.__version__)

    print("Train dir: {0}".format(train_dir))
    print("Test dir: {0}".format(test_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the acoustic language-id algorithms.')
    parser.add_argument('--train-dir', dest='train_dir', action='store', required=True, help='Training directory')
    parser.add_argument('--test-dir', dest='test_dir', action='store', required=True, help='Testing directory')
    args = parser.parse_args()
    sys.exit(main(args))