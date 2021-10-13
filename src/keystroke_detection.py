# keystroke detection using fft and normalization
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct

from scipy import stats
from scipy import signal

import math

import matplotlib
import matplotlib.pyplot as plt

from python_speech_features import mfcc
import librosa

from sklearn import metrics

import json
import pprint
import pickle

def draw(x, _linewidth=None, _color=None):
    plt.plot(x, linewidth=_linewidth, color=_color)
    plt.show()


def rms(series):
    return math.sqrt(sum(series ** 2) / series.size)


def normalize(series):
    return series / rms(series)


def are_frames_intersecting(f1, f2):
    return not ( f1[0] >= f2[1] or f2[0] >= f1[1] )

# frames : (start, end)
def nms(powers, frames):

    packed = list(enumerate(zip(powers, frames))) # (index, (power, (frame_start, frame_end)))

    packed_power_ordered = sorted(packed, key=lambda x: x[1][0], reverse=True)

    # processed means either suppressed or added to the resulting list due to being the max around
    processed = set() 

    # the incides of those that are not suppressed
    max_index = set()

    # from the one with the highest prob to the lowest
    for i, (power, frame) in packed_power_ordered:
        if i in processed:
            continue
        
        # newly processing
        processed.add(i)

        # add to the result - this is the max around since it is not processed earlier
        max_index.add(i)

        # remove the intersecting results as this one has the highest confidence
        # removal is done implicity by adding those windows to processed set
        for i2, (_, frame2) in packed_power_ordered:
            # if any other data intersects with the max found, make it processed to make sure 
            # it is suppressed (not added to the result later)
            if are_frames_intersecting(frame, frame2):
                processed.add(i2)
    
    result = []

    for i, h in enumerate(powers):
        if i in max_index:
            result.append(h)
        else:
            result.append(0)

    # put in index order
    return result


def read_gt_moments(gt_sync, filename="../log.txt"):
    # Actual moments
    gt = []
    keys = []

    # Opening JSON file 
    f = open(filename,'r') 
    
    # returns JSON object as a dictionary 
    data = json.load(f) 

    for i in data:
        gt.append(16*(int(data[i]['time']) - gt_sync))
        keys.append(data[i]["key"])

    return gt, keys



# hypo: 0s and 1s, len: windows
# midpoints: the midpoints of windows
# keytap_moments: the midpoints of keytap moments

# associate positive predictions with the ground truths, such that, we would have 
# a list of elements (hypo index, ground truth index, IoU)
def associate_positive_preds_with_groundtruthts(hypo, midpoints, keytap_moments, window_len, true_positive_IoU_threshold):
    associations = []
    
    for hypo_ind, hm in enumerate(zip(hypo, midpoints)): # each hypo is associated with a midpoint
        h, m = hm
        # if the hypo is a positive, we will need an association for that
        if h == 1:
            # search the key tap moment that is closest to the hypo
            distances = list(enumerate( [ abs(m - keytap_moment) for keytap_moment in keytap_moments ] ))

            # find the closest
            min_dist = min(distances, key=lambda x: x[1])
            keytap_moment_index, min_dist = min_dist

            # compute the IoU
            intersection = 0 if min_dist > (window_len / 2) else ((window_len / 2) - min_dist) / (window_len / 2)

            # check if it meets the threshold
            if intersection >= true_positive_IoU_threshold:
                # add to the list of associations
                association = (hypo_ind, keytap_moment_index, intersection)
                associations.append(association)

    # go over all associations. if there is more than one association for a groundtruth, keep the one with the maximum IoU
    associations.sort(key=lambda x: x[2]) # sort about IoU

    association_to_remove_hypo_indices = set()
    associated_gt_index = set()

    for association in associations:
        if association[1] not in associated_gt_index:
            # the association for gt that has the highest IoU for that gt
            associated_gt_index.add(association[1])
            continue
        else:
            # to be removed from the list
            association_to_remove_hypo_indices.add(association[0])

    old_associations = associations
    associations = []

    for assocation in old_associations:
        if assocation[0] not in association_to_remove_hypo_indices:
            associations.append(association)
    
    return associations



def evaluate(labels, frame_boundaries, gt_moments, true_positive_IoU_thresholds=[ 0.25, 0.5]):
    
    window_len = abs(frame_boundaries[0][0] - frame_boundaries[0][1])
    window_mid_point = [ (w[0] + w[1]) // 2  for w in frame_boundaries]

    # Each element is a dictionary with keys: "classification_threshold", "tp", "fp", "tn", "fn"
    results = []


    for IoU_threshold in true_positive_IoU_thresholds:
        result = { 'IoU_threshold' : IoU_threshold }
        
        associations = associate_positive_preds_with_groundtruthts(labels, window_mid_point, gt_moments, window_len, IoU_threshold)
        
        predicted_positives_count = sum(labels)
        predicted_negatives_count = len(labels) - predicted_positives_count

        gt_positives_count = len(gt_moments)
        gt_negatives_count = len(window_mid_point) - gt_positives_count

        true_positive_count = len(associations)
        false_positive_count = predicted_positives_count - true_positive_count

        false_negative_count = gt_positives_count - true_positive_count
        true_negative_count = predicted_negatives_count - false_negative_count

        # for using sklearn, create y_pred and y_score that would be compatible with the results we found
        y_true = []
        y_pred = []

        # true positives
        y_true.extend( [1] * true_positive_count )
        y_pred.extend( [1] * true_positive_count )

        # false positives
        y_true.extend( [0] * false_positive_count )
        y_pred.extend( [1] * false_positive_count )
            
        # true negatives
        y_true.extend( [0] * true_negative_count )
        y_pred.extend( [0] * true_negative_count )

        # false negatives
        y_true.extend( [1] * false_negative_count )
        y_pred.extend( [0] * false_negative_count )

        confusion_matrix = metrics.confusion_matrix( y_true, y_pred, [0, 1])

        tn, fp, fn, tp = confusion_matrix.ravel()

        # Fill the result
        result["tp"] = true_positive_count
        result["fp"] = false_positive_count
        result["tn"] = true_negative_count
        result["fn"] = false_negative_count
        result["tpr"] = true_positive_count / (true_positive_count + false_negative_count)
        result["tnr"] = true_negative_count / (true_negative_count + false_positive_count)
        result["fpr"] = false_positive_count / (false_positive_count + true_negative_count)
        result["fnr"] = false_negative_count / (false_negative_count + true_positive_count)
        result["acc"] = (true_positive_count + true_negative_count) / (true_positive_count + true_negative_count + false_positive_count + false_negative_count)

        if (true_positive_count + false_positive_count) != 0:
            result["precision"] = true_positive_count / (true_positive_count + false_positive_count)
        else:
            result["precision"] = 1

        if (true_positive_count + false_negative_count) != 0:
            result["recall"] = true_positive_count / (true_positive_count + false_negative_count)
        else:
            result["recall"] = 1

        # Find the explicit points
        true_positives_hypo_indices = set()
        true_positives_gt_indices = set()
            
        for hypo_index, gt_index, IoU in associations:
            true_positives_hypo_indices.add(hypo_index)
            true_positives_gt_indices.add(gt_index)
            
        # Group true positives and false negatives in gt
        true_positives_gt_points = []
        false_negatives_gt_points = []
        for i, keytap_moment in enumerate(gt_moments):
            if i in true_positives_gt_indices:
                true_positives_gt_points.append(keytap_moment)
            else:
                false_negatives_gt_points.append(keytap_moment)
            
        # Group true positives and false positives in hypo
        true_positives_hypo_points = []
        false_positives_hypo_points = []
        for i, (midpoint, hypo_label) in enumerate(zip(window_mid_point, labels)):
            if hypo_label == 1:
                if i in true_positives_hypo_indices:
                    true_positives_hypo_points.append(midpoint)
                else:
                    false_positives_hypo_points.append(midpoint)

        results.append(result)

    return results



def trim_keytap_moments(signal, frame_boundaries, labels, sample_rate, path, _draw=False, _savewav=False, _gt=False):
    
    indexes = np.nonzero(labels)
    frames_with_keytap = []
    for i in indexes[0]:
        frames_with_keytap.append(frame_boundaries[i])

    subsignals = []

    i = 0
    for start, end in frames_with_keytap:
        # crop the signal
        if start < 0:
            start = 0

        subsignal = signal[(start):(end)]

        subsignals.append(subsignal)

        if _draw:
            draw(subsignal)

        if _savewav:
            filename = path + "extracted/keytap{0}.wav".format(i)
            scipy.io.wavfile.write(filename, sample_rate, subsignal)
            i += 1

    subsignals = np.array(subsignals)
    pickle.dump(subsignals, open(path + "X_events", "wb"))



def anything_in_between(l, lower_limit, upper_limit):
    count = 0
    # l is sorted.
    for value in l:
        if value > lower_limit and value < upper_limit:
            count += 1
        if value > upper_limit:
            return count
    return 0


# Distance in ms scale
def evaluate2(gt, pred, margin):
    tp = 0
    fp = 0
    fn = 0
    # for each gt moment, look for a match
    for i, peak_p in enumerate(gt):
        c = anything_in_between(pred, peak_p-margin, peak_p+3*margin)
        if c > 0:
            tp += 1
        else:
            fn += 1
    fp = len(pred) - tp

    print ("GT: {}".format(len(gt)))
    print ("HYPO: {}".format(len(pred)))
    print ("TP: {} FP: {} FN: {}".format(tp, fp, fn))

    return tp, fp, fn


###############################################################################################
###############################################################################################

def main():
    ### CONFIGURATION ####
    path = "../data/BMK/Smartwatch/Subject1/"
    _evaluate = True

    # NOTE: Update this for every recording
    gt_sync = 3177 - 430
    filename = path + 'test/clean/t0.wav'
    ### CONFIGURATION ####

    window_size = 200

    # Read the recording.
    sample_rate, sig = scipy.io.wavfile.read(filename)

    rem = len(sig) % int(sample_rate/100)
    sig = np.array(sig[:len(sig) - rem])

    # at least 200 ms between two keystroke
    minimum_interval = 3200 # 200 * 16khz
    
    sample_length = int((sample_rate * window_size) / 1000)

    # 10 ms
    detection_window_size = 160
    peaks = []
    for x in range(0, len(sig) - detection_window_size):
        peaks.append(np.sum(np.absolute(np.fft.fft(sig[x:x + detection_window_size]))))

    peaks = np.array(peaks)
    tau = np.percentile(peaks, 95)

    x = 0
    events = []
    step = 1
    past_x = - minimum_interval - step
    frame_boundaries = []
    labels = []
    while x < peaks.size:

        before = 0 #int(sample_length / 5)
        after = sample_length - before
        frame_boundaries.append(x)

        if peaks[x] >= tau:
            if x - past_x >= minimum_interval:
                
                # It is a keypress event (maybe)
                keypress = normalize(sig[x:x + sample_length])
                past_x = x

                events.append(keypress)
                labels.append(1)
            x = past_x + minimum_interval
        else:
            x += step
            labels.append(0)

    print ("Number of keystrokes found:", len(events))

    # Evaluate
    if _evaluate:
        gt_moments, keys = read_gt_moments(gt_sync, filename=path+'test/logs/t0.txt')
        result = evaluate2(gt_moments, frame_boundaries, 50)
        pprint.pprint (result)

    # Trim the found keytap moments.
    trim_keytap_moments(sig, frame_boundaries, labels, sample_rate, path)

    ## KEYSTROKE DETECTION END ##


###############################################################################################
###############################################################################################


if __name__ == '__main__':
    main()


###############################################################################################
###############################################################################################
