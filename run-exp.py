import librosa
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import soundfile as sf
from tensorflow.keras import models, layers

# FEAT_LEN = 431  # for 10 seconds of audio, we get 431 MFCC vectors
FEAT_DIM = 40  # each MFCC vector is of 40 dimensions

def load_train_data(dir, locales, limit_per_locale):
    '''
    Load the training data from a directory (FLAC files)
    '''
    sample_file = os.path.join(dir, os.listdir(dir)[0])
    data, samplerate = sf.read(sample_file)
    feat = librosa.feature.mfcc(data, sr=samplerate, n_mfcc=FEAT_DIM)
    feat_len = feat.shape[1]

    X = np.zeros([len(locales) * limit_per_locale, feat_len, FEAT_DIM])
    Y = np.zeros([len(locales) * limit_per_locale, len(locales)])

    locales_count = {x:0 for x in locales}
    locales_to_idx = dict()
    for i, locale in enumerate(locales):
        locales_to_idx[locale] = i

    idx = 0
    for f in os.listdir(dir):
        f_locale = f[:2]
        if locales_count[f_locale] >= limit_per_locale:
            continue

        fpath = os.path.join(dir, f)
        print("Loading {0} ... ".format(fpath))
        data, samplerate = sf.read(fpath)
        feat = librosa.feature.mfcc(data, sr=samplerate, n_mfcc=FEAT_DIM)
        X[idx, :, :] = feat.T
        Y[idx, locales_to_idx[f_locale]] = 1
        idx += 1
        locales_count[f_locale] += 1
    return X, Y, locales_to_idx, feat_len


def create_rnn_model(input_layer_dim, hidden_layer_dims, output_layer_dim):
    model = models.Sequential()
    model.add(layers.LSTM(input_layer_dim, return_sequences=True, unroll=False))
    for idx, layer_dim in enumerate(hidden_layer_dims):
        if idx == len(hidden_layer_dims) - 1:
            return_seq = False
        else:
            return_seq = True
        model.add(layers.LSTM(layer_dim, return_sequences=return_seq))
    model.add(layers.Dense(output_layer_dim, activation='softmax'))
    # model.add(layers.Softmax(num_outputs))
    return model


def main(args):
    train_dir = args.train_dir
    test_dir = args.test_dir
    batch_size = 32
    epochs = 3
    model_file = r"C:\Users\menphix\Documents\workspace\build\final-project\models\rnn_model.pb"
    print(keras.__version__)

    print("Train dir: {0}".format(train_dir))
    print("Test dir: {0}".format(test_dir))
    locales = ['en', 'de', 'es']

    # train_X shape: [batch, timesteps, feature_dim]
    # train_X, train_Y, locales_to_idx, train_feat_len = load_train_data(train_dir, locales, 512)
    # model = create_rnn_model(100, [100], 3)
    # print(train_X.shape)
    # print(train_Y.shape)
    # print(test_X.shape)
    # print(test_Y.shape)
    # print(train_Y)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs)
    # print(history)
    # model.save(model_file)

    test_X, test_Y, _, test_feat_len = load_train_data(test_dir, locales, 30)
    model = models.load_model(model_file)
    predictions = model.predict_classes(test_X)
    print(predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the acoustic language-id algorithms.')
    parser.add_argument('--train-dir', dest='train_dir', action='store', required=True, help='Training directory')
    parser.add_argument('--test-dir', dest='test_dir', action='store', required=True, help='Testing directory')
    args = parser.parse_args()
    sys.exit(main(args))