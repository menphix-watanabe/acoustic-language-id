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


def load_train_data(dir, locales, timesteps, limit_per_locale, channel_last=False):
    '''
    Load the training data from a directory (FLAC files)
    '''
    # sample_file = os.path.join(dir, os.listdir(dir)[0])
    # data, samplerate = sf.read(sample_file)
    # feat = librosa.feature.mfcc(data, sr=samplerate, n_mfcc=FEAT_DIM)
    # feat_len = feat.shape[1]

    X = np.zeros([len(locales) * limit_per_locale, timesteps, FEAT_DIM])
    Y_onehot = np.zeros([len(locales) * limit_per_locale, len(locales)])
    Y = np.zeros(len(locales) * limit_per_locale)

    locales_count = {x: 0 for x in locales}
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

        # Limiting both X and feat to have timesteps in length
        feat = feat.T
        feat_timesteps = feat.shape[0]

        # if feat_timesteps > timesteps:
        # feat_timesteps = timesteps
        if feat_timesteps >= timesteps:
            X[idx, :timesteps, :] = feat[:timesteps, :]
        else:
            # feat_timesteps < timesteps
            remainder = timesteps
            ridx = 0
            while feat_timesteps <= remainder:
                X[idx, ridx * feat_timesteps:(ridx + 1) * feat_timesteps, :] = feat
                ridx += 1
                remainder -= feat_timesteps
            X[idx, ridx * feat_timesteps:, :] = feat[:remainder, :]

            # X[idx, :feat_timesteps, :] = feat.T[:feat_timesteps, :]
            # remainder = timesteps - feat_timesteps
            # while remainder > timesteps:
            #     X[idx, ]
        Y_onehot[idx, locales_to_idx[f_locale]] = 1
        Y[idx] = locales_to_idx[f_locale]
        idx += 1
        locales_count[f_locale] += 1

    if channel_last:
        # Expand one dimension in the last axis
        X = np.expand_dims(X, axis=3)

    return X, Y, Y_onehot, locales_to_idx


def create_rnn_model(input_layer_dim, hidden_layer_dims, output_layer_dim):
    model = models.Sequential()
    model.add(layers.LSTM(input_layer_dim, return_sequences=True, unroll=False))
    for idx, layer_dim in enumerate(hidden_layer_dims):
        if idx == len(hidden_layer_dims) - 1:
            return_seq = False
        else:
            return_seq = True
        model.add(layers.LSTM(layer_dim, return_sequences=return_seq))
        model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_layer_dim, activation='softmax'))
    return model


def create_cnn_model(output_layer_dim):
    model = models.Sequential()
    model.add(layers.Conv2D(16, kernel_size=3))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, kernel_size=3))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, kernel_size=3))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(output_layer_dim, activation='softmax'))
    return model


def eval(predictions, truth):
    return predictions[predictions == truth].shape[0] / truth.shape[0]


def get_model_name(model_type, input_layer_dim, hidden_layer_dims, output_layer_dim, train_data, timesteps, epochs):
    return "i{0}_h{1}_o{2}_{3}_samples_{4}_epochs_{5}_timesteps_{6}.pb".format(input_layer_dim, '_'.join(
        [str(x) for x in hidden_layer_dims]), output_layer_dim, train_data, epochs, timesteps, model_type)


def main(args):
    # Training data: 73,080 in total, 24360 per language
    # Testing data: 540 in total, 180 per language
    train_dir = args.train_dir
    test_dir = args.test_dir
    save_model_dir = args.save_model_dir
    load_model_file = args.load_model
    batch_size = 32
    epochs = 5
    timesteps = 512
    input_layer_dim = 100
    hidden_layer_dims = [100]
    output_layer_dim = 3
    num_samples = 24000
    model_type = 'cnn'
    print(keras.__version__)

    print("Train dir: {0}".format(train_dir))
    print("Test dir: {0}".format(test_dir))
    locales = ['en', 'de', 'es']

    # train_X shape: [batch, timesteps, feature_dim]
    if model_type == 'cnn':
        channel_last = True
    else:
        channel_last = False

    # print(test_X.shape)
    # print(test_Y.shape)
    # print(train_Y)

    if load_model_file:
        model = keras.models.load_model(load_model_file)
        test_X, test_Y, test_Y_onehot, _ = load_train_data(test_dir, locales, timesteps, 180, channel_last=channel_last)
        predictions = model.predict_classes(test_X)
        print(predictions)
        print(test_Y)
        accuracy = eval(predictions, test_Y)
        print("Accuracy: {0:.2f}%".format(100 * accuracy))
    else:
        train_X, train_Y, train_Y_onehot, locales_to_idx = load_train_data(train_dir, locales, timesteps, num_samples,
                                                                           channel_last=channel_last)
        print(train_X.shape)
        print(train_Y_onehot.shape)
        print(train_Y.shape)
        # model = create_rnn_model(input_layer_dim, hidden_layer_dims, output_layer_dim)
        model = create_cnn_model(output_layer_dim)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_X, train_Y_onehot, batch_size=batch_size, epochs=epochs)
        print(history)

        if save_model_dir:
            model_file = os.path.join(save_model_dir,
                                      get_model_name(model_type, input_layer_dim, hidden_layer_dims, output_layer_dim,
                                                     num_samples, timesteps, epochs))
            print("Saving model to: {0}".format(model_file))
            model.save(model_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the acoustic language-id algorithms.')
    parser.add_argument('--train-dir', dest='train_dir', action='store', required=True, help='Training directory')
    parser.add_argument('--test-dir', dest='test_dir', action='store', required=True, help='Testing directory')
    parser.add_argument('--save-model-dir', dest='save_model_dir', action='store', required=False,
                        help='Directory to save model')
    parser.add_argument('--load-model', dest='load_model', action='store', required=False,
                        help='Load the model instead of training it')
    # parser.add_argument('--input-layer-dim', dest='input_layer_dim', action='store', type=int, required=True, help='Dimension for the input layer')
    # parser.add_argument('--hidden-layer-dims', dest='hidden_layer_dims', action='store', type=int, required=True,
    #                     help='Dimensions for the hidden layer, separated by comma')
    # parser.add_argument('--time-steps', dest='time_steps', action='store', type=int, required=True,
    #                     help='Number of time steps for RNN. For CNN it\'s the feature\'s length')
    # parser.add_argument('--model-type', dest='model_type', action='store', required=True,
    #                     help='Model type: rnn/cnn')
    # parser.add_argument('--action', dest='action', action='store', required=True,
    #                     help='Action: train/test/test-file')
    # parser.add_argument('--load-model-file', dest='load_model_file', action='store', required=False,
    #                     help='Load the model from this file')
    # parser.add_argument('--save-model-file', dest='save_model_file', action='store', required=False,
    #                     help='Save the model to this file')
    args = parser.parse_args()
    sys.exit(main(args))
