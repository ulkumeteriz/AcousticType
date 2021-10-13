import numpy as np

from scipy.stats import loggamma

label_mapping = {
    "A" : 0,
    "B" : 1,
    "C" : 2,
    "D" : 3,
    "E" : 4,
    "F" : 5,
    "G" : 6,
    "H" : 7,
    "I" : 8,
    "J" : 9,
    "K" : 10,
    "L" : 11,
    "M" : 12,
    "N" : 13,
    "O" : 14,
    "P" : 15,
    "Q" : 16,
    "R" : 17,
    "S" : 18,
    "T" : 19,
    "U" : 20,
    "V" : 21,
    "W" : 22,
    "X" : 23,
    "Y" : 24,
    "Z" : 25,
    "0" : 26,
    "1" : 27,
    "2" : 28,
    "3" : 29,
    "4" : 30,
    "5" : 31,
    "6" : 32,
    "7" : 33,
    "8" : 34,
    "9" : 35,
    0 : "A",
    1 : "B",
    2 : "C",
    3 : "D",
    4 : "E",
    5 : "F",
    6 : "G",
    7 : "H",
    8 : "I",
    9 : "J",
    10 : "K",
    11 : "L",
    12 : "M",
    13 : "N",
    14 : "O",
    15 : "P",
    16 : "Q",
    17 : "R",
    18 : "S",
    19 : "T",
    20 : "U",
    21 : "V",
    22 : "W",
    23 : "X",
    24 : "Y",
    25 : "Z",
    26 : "0",
    27 : "1",
    28 : "2",
    29 : "3",
    30 : "4",
    31 : "5",
    32 : "6",
    33 : "7",
    34 : "8",
    35 : "9"
}

def intrinsic_entropy(num_classes=36):
    p_k = 1.0 / num_classes
    i_entropy = -num_classes * (p_k * np.log2(p_k))
    return i_entropy

def extrinsic_entropy(class_probs):
    e_entropy = 0
    for p_k in class_probs:
        if p_k == 0:
            continue
        e_entropy -= p_k * np.log2(p_k)
    return e_entropy

def information_gain(class_probs):
    ig = intrinsic_entropy(len(class_probs)) - extrinsic_entropy(class_probs)
    return ig

def word_information_gain(class_probs_list):
    ig = 0
    for class_probs in class_probs_list:
        ig += information_gain(class_probs)
    return ig

def guessing_entropy(gt, pred, class_probs):
    total_entropy = 0
    for i, p in enumerate(pred):
        if i >= len(gt):
            print (i)
            break
        # Prediction is correct.
        if p == gt[i]:
            total_entropy += extrinsic_entropy(class_probs[i])
        # Prediction is wrong.
        else:
            total_entropy += intrinsic_entropy()
    return total_entropy / len(gt)

def guessing_info_gain(gt, pred, class_probs):
    g_entropy = guessing_entropy(gt,pred,class_probs)
    actual_entropy = intrinsic_entropy()
    return (actual_entropy - g_entropy)    

def dataset_stats(text):
    counts = {}
    for c in text:
        if c not in counts.keys():
            counts[c] = 1
        else:
            counts[c] += 1
    print (counts)
    print()

def type_stats(text):
    counts = {'alpha': 0, 'num':0}
    for c in text:
        if c.isalpha():
            counts['alpha'] += 1
        else:
            counts['num'] += 1
    print (counts)
    print()

def location_stats(text):
    left_pinky = ['1', 'Q', 'A', 'Z']
    left_ring = ['2', 'W', 'S', 'X']
    left_middle = ['3','E','D', 'C']
    left_index = ['4','5','R','T','F','G','V','B']
    left = left_pinky + left_ring + left_middle + left_index
    right_index = ['6','7','Y','U','H','J','N','M']
    right_middle =['8','I','K']
    right_ring = ['9','O','L']
    right_pinky = ['0','P']
    right = right_index + right_middle + right_ring + right_pinky

    counts = {'left': 0, 'left_index': 0, 'left_middle': 0,'left_ring': 0,'left_pinky': 0, \
        'right': 0, 'right_index': 0,'right_middle': 0,'right_ring': 0, 'right_pinky': 0}

    for c in text:
        if c in left:
            counts['left'] += 1
            if c in left_index:
                counts['left_index'] += 1
            elif c in left_middle:
                counts['left_middle'] += 1
            elif c in left_ring:
                counts['left_ring'] += 1
            elif c in left_pinky:
                counts['left_pinky'] += 1
        elif c in right:
            counts['right'] += 1
            if c in right_index:
                counts['right_index'] += 1
            elif c in right_middle:
                counts['right_middle'] += 1
            elif c in right_ring:
                counts['right_ring'] += 1
            elif c in right_pinky:
                counts['right_pinky'] += 1
    print (counts)

def stats():
    print ('Random Password')
    p0 = ['M58NGMGYKZ83NCE2', 'A9ZUCPZKXB8KAM3L', 'QKPKMLZ9DU9AEZ94', '5VYM3T3D2B9F67BC', 'UEAU6RDARY7VDA4R', '6SCW7RW4V8GBHHBK', 'EA86BA5H7G8L3F5X', 'J3ZS26DXTCSGJZLX', 'QC8P3J2UJTZPWXKP', 'ZG9C6LQ8URFRSHKP', 'KNCZAMD2M6BMNAJN', 'X6HNLKF92VPUWMHW', '3FNNYKXWPM54697Z', '9TX8FZQHNS43AYHH', 'V5N8L2WKL4989E85', '7WGK7DABRLQLNS9T']
    text = ''.join(p0)
    dataset_stats(text)
    type_stats(text)
    location_stats(text)

    print ('Selected Password')
    p1 = ['123456', 'ILOVEYOU', 'TINKERBELL', 'BABYGIRL9', 'CLASSOF2008', '070789', 'Q1W2E3R4T5', 'TRYAGAIN', 'SMELLY1', 'CAREBEARS1', 'I10VEY0U', 'PASS1WORD', 'PANICATTHEDISCO', '2CHILDREN', 'COUNTRYGIRL', 'HEARTAGRAM', '123QWEASD', 'BITEME2', 'DANIEL13', 'SOMETHING1']
    text = ''.join(p1)
    dataset_stats(text)
    type_stats(text)
    location_stats(text)

    print ('Mail')
    mail = ['THANKYOUFORATTENDINGTHECALLITWAS', 'REALLYNICETOLEARNABOUTYOU', 'ANDYOURBUSINESSIHAVEATTACHED', 'MYCUSTOMPLANFORYOURBUSINESSTO', 'THISEMAILPLEASEGOTHROUGHITAND', 'LETMEKNOWIFYOUHAVEANYCONCERNS']
    text = ''.join(mail)
    dataset_stats(text)
    type_stats(text)
    location_stats(text)

    print ('Overall')
    text = ''.join(mail + p0 + p1)
    dataset_stats(text)
    type_stats(text)
    location_stats(text)

