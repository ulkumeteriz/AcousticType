import json
import pickle
import numpy as np
import scipy.io.wavfile

import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from python_speech_features import mfcc

from utility import label_mapping


def draw(x):
    plt.plot(x)
    plt.show()

class Conf:
    def __init__(self, path, train_count, test_count, train_log_t, train_record_t, test_log_t, test_record_t):
        self.path = path
        self.train_count = train_count
        self.test_count = test_count
        # get ready the training data configurations
        f = ["s{}".format(i) for i in range(train_count)]
        self.train_records = ['{}/train/clean/{}.wav'.format(self.path, i) for i in f]
        self.train_logs = ['{}/train/logs/{}.txt'.format(self.path, i) for i in f]
        self.train_gt_syncs = [train_log_t[i] - train_record_t[i] for i in range(train_count)]
        self.train_X_filename = "{}/X_train".format(self.path)
        self.train_Y_filename = "{}/Y_train".format(self.path)

        f = ["t{}".format(i) for i in range(test_count)]
        self.test_records = ['{}/test/clean/{}.wav'.format(self.path, i) for i in f]
        self.test_logs = ['{}/test/logs/{}.txt'.format(self.path, i) for i in f]
        self.test_gt_syncs = [test_log_t[i] - test_record_t[i] for i in range(test_count)]
        self.test_X_filename = "{}/X_test".format(self.path)
        self.test_Y_filename = "{}/Y_test".format(self.path)

    def read_gt_moments(self, gt_sync, filename):
        # Actual moments
        gt = []

        labels = []

        # Opening JSON file 
        f = open(filename,'r') 
        
        # returns JSON object as a dictionary 
        data = json.load(f) 

        # sort the keys
        sorted_keys = sorted( [ int(d) for d in data.keys() ] )

        for i in sorted_keys:
            gt.append( 16 * (int(data[str(i)]['time']) - gt_sync))
            labels.append( data[str(i)]["key"] )

        return gt, labels

    def extract_keytap_moments(self, signal, windows, labels, frame_per_ms, _draw=False):
        moments = []

        for i, (start, end) in enumerate(windows):
            # crop the signal        
            subsignal = signal[start:end]

            if len(subsignal) == 0:
                continue

            moments.append(subsignal)

            if _draw and ( i < 5 or i > len(windows) - 5):
                print(start, end)
                print (labels[i])
                draw(subsignal)

        return np.array(moments)


    # Reads all training and test data 
    def preprocess(self, window_length=200):

        # For each training record...
        for index, record_filename in enumerate(self.train_records):
            log_filename = self.train_logs[index]
            gt_sync = self.train_gt_syncs[index]

            # Read the audio file
            sample_rate, sig = scipy.io.wavfile.read(record_filename)

            # Set the configurations
            frame_per_ms = sample_rate / 1000
            window_size = window_length * frame_per_ms 

            # Read the log file
            gt, labels = self.read_gt_moments(gt_sync, log_filename)

            # Set the margin
            s = int(window_size/4)

            # Set the indexes of the window boundaries
            gt_windows = [ [ max(0,moment - s), (max(0,moment-s) + int(window_size))] for moment in gt]

            # Get the windows and corresponding labels
            moments = self.extract_keytap_moments(sig, gt_windows, labels, frame_per_ms)

            if index == 0:
                raw_X_train = moments
                Y_train = labels
            else:
                raw_X_train = np.concatenate((raw_X_train, moments))
                Y_train = np.concatenate((Y_train, labels))

        # For each test record...
        for index, record_filename in enumerate(self.test_records):

            log_filename = self.test_logs[index]
            gt_sync = self.test_gt_syncs[index]

            # Read the audio file
            sample_rate, sig = scipy.io.wavfile.read(record_filename)

            # Set the configurations
            frame_per_ms = sample_rate / 1000
            window_size = window_length * frame_per_ms 

            # Read the log file
            gt, labels = self.read_gt_moments(gt_sync, log_filename)
            
            # Set the margin
            s = int(window_size/4)

            # Set the indexes of the window boundaries
            gt_windows = [ [ max(0,moment - s), (max(0,moment-s) + int(window_size))] for moment in gt]

            # Get the windows and corresponding labels
            moments = self.extract_keytap_moments(sig, gt_windows, labels, frame_per_ms, _draw=False)

            if index == 0: 
                raw_X_test = moments
                Y_test = labels
            else:
                raw_X_test = np.concatenate((raw_X_test,moments))
                Y_test = np.concatenate((Y_test,labels))

        # Clean the redundant characters in training data
        redundant_indices = np.where((Y_train == ' ') | (Y_train == 'Shift') | (Y_train == 'Backspace') | (Y_train == ',') | (Y_train == 'CapsLock') )
        raw_X_train = np.delete(raw_X_train, redundant_indices, axis=0)
        Y_train_chars = np.delete(Y_train, redundant_indices)

        ###########
        # RAW DATA FOR CROSS-CORRELATION SIMILARITY SCORE AMONG MOST CONFUSED CLASSES

        # def dump_raw(labels):
        #     for label in labels:
        #         indices = np.where((Y_train == label))
        #         pickle.dump(raw_X_train[indices], open(self.path + "/" + label, "wb" ))
        
        # # Most confused labels
        # dump_raw("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

        # return
        ###########

        # Prepare training labels by converting letters into numbers
        y = []
        for l in Y_train_chars:
            yy = label_mapping[l]
            y.append(yy)
        Y_train = np.array(y)

        # Clean the redundant characters in test data
        redundant_indices = np.where((Y_test == ' ') | (Y_test == 'Shift') | (Y_test == 'Backspace') | (Y_test == ',') | (Y_test == 'CapsLock') )
        raw_X_test = np.delete(raw_X_test, redundant_indices, axis=0)
        Y_test = np.delete(Y_test, redundant_indices)


        # sample rate of the recordings
        rate = 16000

        # Get MFCC features of training data
        X_train = []
        for signal in raw_X_train:
            signal = signal.astype(float)
            mel = mfcc(signal, samplerate=rate,
                    numcep=64, nfilt=64, nfft=1103).T
            X_train.append(mel)

        # Unscaled training data ready in a numpy array
        X_train = np.array(X_train)

        ###########
        # DATA FOR CROSS-CORRELATION SIMILARITY SCORE AMONG MOST CONFUSED CLASSES

        # def dump_raw(labels):
        #     for label in labels:
        #         indices = np.where((Y_train_chars == label))
        #         pickle.dump(X_train[indices], open(self.path + "/confused/" + label, "wb" ))
        
        # # Most confused labels
        # dump_raw("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

        # return
        ###########


        # Get the mfcc features of test data
        X_test = []
        for signal in raw_X_test:
            signal = signal.astype(float)
            mel = mfcc(signal, samplerate=rate,
                    numcep=64, nfilt=64, nfft=1103).T
            X_test.append(mel)
        
        # Unscaled test data ready in a numpy array
        X_test = np.array(X_test)

        # Scale all data...
        scale = np.concatenate((X_train, X_test), axis=0)

        scalers = {}
        for i in range(scale.shape[1]):
            scalers[i] = MinMaxScaler()
            # fit the scaler on all data
            scalers[i].fit(scale[:, i, :])
            # scale training data
            X_train[:, i, :] = scalers[i].transform(X_train[:, i, :])
            # scale test data
            X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])

        # reshape the data
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

        print ("Train shape:", X_train.shape, Y_train.shape)
        print ("Test shape:", X_test.shape, Y_test.shape)

        # dump the scaled training and test data
        pickle.dump(X_train, open(self.train_X_filename, "wb" ))
        pickle.dump(Y_train, open(self.train_Y_filename, "wb" )) 
        pickle.dump(X_test, open(self.test_X_filename, "wb" ))
        pickle.dump(Y_test, open(self.test_Y_filename, "wb" ))

############################################################################################################

# NOTE Synchronizing the recordings with the logs is required when a new data is added.
# X_log_t : the starting time - the time stamp of the first key - in the logs 
# X_record_t : the time (in ms) of the first keystroke sound in the recordings

def main_MBA():
    train_log_t = [7061, 7236, 3197, 4589, 3418]
    train_record_t = [860, 320, 800, 1360, 1600]

    test_log_t = [4531, 3613, 5993]
    test_record_t = [1280,830,2250]

    conf = Conf('../data/MBA/Subject1', 5, 3, train_log_t, train_record_t, test_log_t, test_record_t)
    
    conf.preprocess()

def main_BMK():
    train_log_t = [4292, 3144, 3174, 4038, 6422]
    train_record_t = [1120, 747, 750, 1070, 3030]

    test_log_t = [3876, 4598, 3891]
    test_record_t = [1414, 2036, 1630]

    conf = Conf('../data/BMK/Subject1', 5, 3, train_log_t, train_record_t, test_log_t, test_record_t)
    
    conf.preprocess()
