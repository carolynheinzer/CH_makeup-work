import os
import csv
import pandas

import scipy
from scipy.signal import find_peaks

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def to_dict(file_name: str, code: int):
    # return dict of attributes and their values

    if code == 0:
        name = "Data/Lab3/Labeled/" + file_name
    if code == 1:
        name = "Data/Lab3/Adversary/" + file_name

    f = open(name, "r")

    # get keys
    info = {}
    firstline = f.readline()

    # remove newlines
    split = firstline.split(",")
    keys = [key.replace("\n", "") for key in split]
    keys = [key.replace(".", "_") for key in keys]
    # print(keys)
    # print(len(keys))
    
    for key in keys:
        info[key] = []

    # put into dict
    for line in f.readlines():
        split = line.split(",")
        att_vals = [val.replace("\n", "") for val in split]
        # print(att_vals)

        for i in range(0, 37):
            el = att_vals[i]
            info[keys[i]].append(float(el))

    f.close()
    return info

# print(to_dict("Data/Lab3/Train/CAR_01.csv"))

def preprocess(file_name: str, code: int):
    # mean, std, number of peaks
    info = to_dict(file_name, code)

    # initialize results var
    res = {}

    headset_vel_x = list(map(float, info['headset_vel_x']))
    headset_vel_y = list(map(float, info['headset_vel_y']))
    headset_vel_z = list(map(float, info['headset_vel_z']))
    headset_angularVel_x = list(map(float, info['headset_angularVel_x']))
    headset_angularVel_y = list(map(float, info['headset_angularVel_y']))
    headset_angularVel_z = list(map(float, info['headset_angularVel_z']))
    headset_pos_x = list(map(float, info['headset_pos_x']))
    headset_pos_y = list(map(float, info['headset_pos_y']))
    headset_pos_z = list(map(float, info['headset_pos_z']))
    headset_rot_x = list(map(float, info['headset_rot_x']))
    headset_rot_y = list(map(float, info['headset_rot_y']))
    headset_rot_z = list(map(float, info['headset_rot_z']))
    controller_left_vel_x = list(map(float, info['controller_left_vel_x']))
    controller_left_vel_y = list(map(float, info['controller_left_vel_y']))
    controller_left_vel_z = list(map(float, info['controller_left_vel_z']))
    controller_left_angularVel_x = list(map(float, info['controller_left_angularVel_x']))
    controller_left_angularVel_y = list(map(float, info['controller_left_angularVel_y']))
    controller_left_angularVel_z = list(map(float, info['controller_left_angularVel_z']))
    controller_left_pos_x = list(map(float, info['controller_left_pos_x']))
    controller_left_pos_y = list(map(float, info['controller_left_pos_y']))
    controller_left_pos_z = list(map(float, info['controller_left_pos_z']))
    controller_left_rot_x = list(map(float, info['controller_left_rot_x']))
    controller_left_rot_y = list(map(float, info['controller_left_rot_y']))
    controller_left_rot_z = list(map(float, info['controller_left_rot_z']))
    controller_right_vel_x = list(map(float, info['controller_right_vel_x']))
    controller_right_vel_y = list(map(float, info['controller_right_vel_y']))
    controller_right_vel_z = list(map(float, info['controller_right_vel_z']))
    controller_right_angularVel_x = list(map(float, info['controller_right_angularVel_x']))
    controller_right_angularVel_y = list(map(float, info['controller_right_angularVel_y']))
    controller_right_angularVel_z = list(map(float, info['controller_right_angularVel_z']))
    controller_right_pos_x = list(map(float, info['controller_right_pos_x']))
    controller_right_pos_y = list(map(float, info['controller_right_pos_y']))
    controller_right_pos_z = list(map(float, info['controller_right_pos_z']))
    controller_right_rot_x = list(map(float, info['controller_right_rot_x']))
    controller_right_rot_y = list(map(float, info['controller_right_rot_y']))
    controller_right_rot_z = list(map(float, info['controller_right_rot_z']))

    # find the numbers of peaks in these sets of y values
    peaks_headset_vel_x, _ = find_peaks(headset_vel_x)
    peaks_headset_vel_y, _ = find_peaks(headset_vel_y)
    peaks_headset_vel_z, _ = find_peaks(headset_vel_z)
    peaks_headset_rot_x, _ = find_peaks(headset_vel_x)
    peaks_headset_rot_y, _ = find_peaks(headset_vel_y)
    peaks_headset_rot_z, _ = find_peaks(headset_vel_z)
    peaks_headset_angularVel_x, _ = find_peaks(headset_angularVel_x)
    peaks_headset_angularVel_y, _ = find_peaks(headset_angularVel_y)
    peaks_headset_angularVel_z, _ = find_peaks(headset_angularVel_z)
    peaks_controller_left_vel_x, _ = find_peaks(controller_left_vel_x)
    peaks_controller_left_vel_y, _ = find_peaks(controller_left_vel_y)
    peaks_controller_left_vel_z, _ = find_peaks(controller_left_vel_z)
    peaks_controller_left_angularVel_x, _ = find_peaks(controller_left_angularVel_x)
    peaks_controller_left_angularVel_y, _ = find_peaks(controller_left_angularVel_y)
    peaks_controller_left_angularVel_z, _ = find_peaks(controller_left_angularVel_z)
    peaks_controller_left_pos_x, _ = find_peaks(controller_left_pos_x)
    peaks_controller_left_pos_y, _ = find_peaks(controller_left_pos_y)
    peaks_controller_left_pos_z, _ = find_peaks(controller_left_pos_z)
    peaks_controller_left_rot_x, _ = find_peaks(controller_left_rot_x)
    peaks_controller_left_rot_y, _ = find_peaks(controller_left_rot_y)
    peaks_controller_left_rot_z, _ = find_peaks(controller_left_rot_z)
    peaks_controller_right_vel_x, _ = find_peaks(controller_right_vel_x)
    peaks_controller_right_vel_y, _ = find_peaks(controller_right_vel_y)
    peaks_controller_right_vel_z, _ = find_peaks(controller_right_vel_z)
    peaks_controller_right_angularVel_x, _ = find_peaks(controller_right_angularVel_x)
    peaks_controller_right_angularVel_y, _ = find_peaks(controller_right_angularVel_y)
    peaks_controller_right_angularVel_z, _ = find_peaks(controller_right_angularVel_z)
    peaks_controller_right_pos_x, _ = find_peaks(controller_right_pos_x)
    peaks_controller_right_pos_y, _ = find_peaks(controller_right_pos_y)
    peaks_controller_right_pos_z, _ = find_peaks(controller_right_pos_z)
    peaks_controller_right_rot_x, _ = find_peaks(controller_right_rot_x)
    peaks_controller_right_rot_y, _ = find_peaks(controller_right_rot_y)
    peaks_controller_right_rot_z, _ = find_peaks(controller_right_rot_z)

    # populate the results dictionary with the number of peaks
    res['peaks_headset_vel_x'] = len(peaks_headset_vel_x)
    res['peaks_headset_vel_y'] = len(peaks_headset_vel_y)
    res['peaks_headset_vel_z'] = len(peaks_headset_vel_z)
    res['peaks_headset_rot_x'] = len(peaks_headset_rot_x)
    res['peaks_headset_rot_y'] = len(peaks_headset_rot_y)
    res['peaks_headset_rot_z'] = len(peaks_headset_rot_z)
    res['peaks_headset_angularVel_x'] = len(peaks_headset_angularVel_x)
    res['peaks_headset_angularVel_y'] = len(peaks_headset_angularVel_y)
    res['peaks_headset_angularVel_z'] = len(peaks_headset_angularVel_z)
    res['peaks_controller_left_vel_x'] = len(peaks_controller_left_vel_x)
    res['peaks_controller_left_vel_y'] = len(peaks_controller_left_vel_y)
    res['peaks_controller_left_vel_z'] = len(peaks_controller_left_vel_z)
    res['peaks_controller_left_angularVel_x'] = len(peaks_controller_left_angularVel_x)
    res['peaks_controller_left_angularVel_y'] = len(peaks_controller_left_angularVel_y)
    res['peaks_controller_left_angularVel_z'] = len(peaks_controller_left_angularVel_z)
    res['peaks_controller_left_pos_x'] = len(peaks_controller_left_pos_x)
    res['peaks_controller_left_pos_y'] = len(peaks_controller_left_pos_y)
    res['peaks_controller_left_pos_z'] = len(peaks_controller_left_pos_z)
    res['peaks_controller_left_rot_x'] = len(peaks_controller_left_rot_x)
    res['peaks_controller_left_rot_y'] = len(peaks_controller_left_rot_y)
    res['peaks_controller_left_rot_z'] = len(peaks_controller_left_rot_z)
    res['peaks_controller_right_vel_x'] = len(peaks_controller_right_vel_x)
    res['peaks_controller_right_vel_y'] = len(peaks_controller_right_vel_y)
    res['peaks_controller_right_vel_z'] = len(peaks_controller_right_vel_z)
    res['peaks_controller_right_angularVel_x'] = len(peaks_controller_right_angularVel_x)
    res['peaks_controller_right_angularVel_y'] = len(peaks_controller_right_angularVel_y)
    res['peaks_controller_right_angularVel_z'] = len(peaks_controller_right_angularVel_z)
    res['peaks_controller_right_pos_x'] = len(peaks_controller_right_pos_x)
    res['peaks_controller_right_pos_y'] = len(peaks_controller_right_pos_y)
    res['peaks_controller_right_pos_z'] = len(peaks_controller_right_pos_z)
    res['peaks_controller_right_rot_x'] = len(peaks_controller_right_rot_x)
    res['peaks_controller_right_rot_y'] = len(peaks_controller_right_rot_y)
    res['peaks_controller_right_rot_z'] = len(peaks_controller_right_rot_z)

    # find mean of all attributes

    # create the series
    s_headset_vel_x = pandas.Series(data = headset_vel_x)
    s_headset_vel_y = pandas.Series(data = headset_vel_y)
    s_headset_vel_z = pandas.Series(data = headset_vel_z)
    s_headset_angularVel_x = pandas.Series(data = headset_angularVel_x)
    s_headset_angularVel_y = pandas.Series(data = headset_angularVel_y)
    s_headset_angularVel_z = pandas.Series(data = headset_angularVel_z)
    s_headset_pos_x = pandas.Series(data = headset_pos_x)
    s_headset_pos_y = pandas.Series(data = headset_pos_y)
    s_headset_pos_z = pandas.Series(data = headset_pos_z)
    s_headset_rot_x = pandas.Series(data = headset_rot_x)
    s_headset_rot_y = pandas.Series(data = headset_rot_y)
    s_headset_rot_z = pandas.Series(data = headset_rot_z)
    s_controller_left_vel_x = pandas.Series(data = controller_left_vel_x)
    s_controller_left_vel_y = pandas.Series(data = controller_left_vel_y)
    s_controller_left_vel_z = pandas.Series(data = controller_left_vel_z)
    s_controller_left_angularVel_x = pandas.Series(data = controller_left_angularVel_x)
    s_controller_left_angularVel_y = pandas.Series(data = controller_left_angularVel_y)
    s_controller_left_angularVel_z = pandas.Series(data = controller_left_angularVel_z)
    s_controller_left_pos_x = pandas.Series(data = controller_left_pos_x)
    s_controller_left_pos_y = pandas.Series(data = controller_left_pos_y)
    s_controller_left_pos_z = pandas.Series(data = controller_left_pos_z)
    s_controller_left_rot_x = pandas.Series(data = controller_left_rot_x)
    s_controller_left_rot_y = pandas.Series(data = controller_left_rot_y)
    s_controller_left_rot_z = pandas.Series(data = controller_left_rot_z)
    s_controller_right_vel_x = pandas.Series(data = controller_right_vel_x)
    s_controller_right_vel_y = pandas.Series(data = controller_right_vel_y)
    s_controller_right_vel_z = pandas.Series(data = controller_right_vel_z)
    s_controller_right_angularVel_x = pandas.Series(data = controller_right_angularVel_x)
    s_controller_right_angularVel_y = pandas.Series(data = controller_right_angularVel_y)
    s_controller_right_angularVel_z = pandas.Series(data = controller_right_angularVel_z)
    s_controller_right_pos_x = pandas.Series(data = controller_right_pos_x)
    s_controller_right_pos_y = pandas.Series(data = controller_right_pos_y)
    s_controller_right_pos_z = pandas.Series(data = controller_right_pos_z)
    s_controller_right_rot_x = pandas.Series(data = controller_right_rot_x)
    s_controller_right_rot_y = pandas.Series(data = controller_right_rot_y)
    s_controller_right_rot_z = pandas.Series(data = controller_right_rot_z)

    # calculate the mean
    res['mean_headset_vel_x'] = s_headset_vel_x.mean()
    res['mean_headset_vel_y'] = s_headset_vel_y.mean()
    res['mean_headset_vel_z'] = s_headset_vel_z.mean()
    res['mean_headset_angularVel_x'] = s_headset_angularVel_x.mean()
    res['mean_headset_angularVel_y'] = s_headset_angularVel_y.mean()
    res['mean_headset_angularVel_z'] = s_headset_angularVel_z.mean()
    res['mean_headset_pos_x'] = s_headset_pos_x.mean()
    res['mean_headset_pos_y'] = s_headset_pos_y.mean()
    res['mean_headset_pos_z'] = s_headset_pos_z.mean()
    res['mean_headset_rot_x'] = s_headset_rot_x.mean()
    res['mean_headset_rot_y'] = s_headset_rot_y.mean()
    res['mean_headset_rot_z'] = s_headset_rot_z.mean()
    res['mean_controller_left_vel_x'] = s_controller_left_vel_x.mean()
    res['mean_controller_left_vel_y'] = s_controller_left_vel_y.mean()
    res['mean_controller_left_vel_z'] = s_controller_left_vel_z.mean()
    res['mean_controller_left_angularVel_x'] = s_controller_left_angularVel_x.mean()
    res['mean_controller_left_angularVel_y'] = s_controller_left_angularVel_y.mean()
    res['mean_controller_left_angularVel_z'] = s_controller_left_angularVel_z.mean()
    res['mean_controller_left_pos_x'] = s_controller_left_pos_x.mean()
    res['mean_controller_left_pos_y'] = s_controller_left_pos_y.mean()
    res['mean_controller_left_pos_z'] = s_controller_left_pos_z.mean()
    res['mean_controller_left_rot_x'] = s_controller_left_rot_x.mean()
    res['mean_controller_left_rot_y'] = s_controller_left_rot_y.mean()
    res['mean_controller_left_rot_z'] = s_controller_left_rot_z.mean()
    res['mean_controller_right_vel_x'] = s_controller_right_vel_x.mean()
    res['mean_controller_right_vel_y'] = s_controller_right_vel_y.mean()
    res['mean_controller_right_vel_z'] = s_controller_right_vel_z.mean()
    res['mean_controller_right_angularVel_x'] = s_controller_right_angularVel_x.mean()
    res['mean_controller_right_angularVel_y'] = s_controller_right_angularVel_y.mean()
    res['mean_controller_right_angularVel_z'] = s_controller_right_angularVel_z.mean()
    res['mean_controller_right_pos_x'] = s_controller_right_pos_x.mean()
    res['mean_controller_right_pos_y'] = s_controller_right_pos_y.mean()
    res['mean_controller_right_pos_z'] = s_controller_right_pos_z.mean()
    res['mean_controller_right_rot_x'] = s_controller_right_rot_x.mean()
    res['mean_controller_right_rot_y'] = s_controller_right_rot_y.mean()
    res['mean_controller_right_rot_z'] = s_controller_right_rot_z.mean()

    # find standard deviation
    res['std_headset_vel_x'] = s_headset_vel_x.std()
    res['std_headset_vel_y'] = s_headset_vel_y.std()
    res['std_headset_vel_z'] = s_headset_vel_z.std()
    res['std_headset_angularVel_x'] = s_headset_angularVel_x.std()
    res['std_headset_angularVel_y'] = s_headset_angularVel_y.std()
    res['std_headset_angularVel_z'] = s_headset_angularVel_z.std()
    res['std_headset_pos_x'] = s_headset_pos_x.std()
    res['std_headset_pos_y'] = s_headset_pos_y.std()
    res['std_headset_pos_z'] = s_headset_pos_z.std()
    res['std_headset_rot_x'] = s_headset_rot_x.std()
    res['std_headset_rot_y'] = s_headset_rot_y.std()
    res['std_headset_rot_z'] = s_headset_rot_z.std()
    res['std_controller_left_vel_x'] = s_controller_left_vel_x.std()
    res['std_controller_left_vel_y'] = s_controller_left_vel_y.std()
    res['std_controller_left_vel_z'] = s_controller_left_vel_z.std()
    res['std_controller_left_angularVel_x'] = s_controller_left_angularVel_x.std()
    res['std_controller_left_angularVel_y'] = s_controller_left_angularVel_y.std()
    res['std_controller_left_angularVel_z'] = s_controller_left_angularVel_z.std()
    res['std_controller_left_pos_x'] = s_controller_left_pos_x.std()
    res['std_controller_left_pos_y'] = s_controller_left_pos_y.std()
    res['std_controller_left_pos_z'] = s_controller_left_pos_z.std()
    res['std_controller_left_rot_x'] = s_controller_left_rot_x.std()
    res['std_controller_left_rot_y'] = s_controller_left_rot_y.std()
    res['std_controller_left_rot_z'] = s_controller_left_rot_z.std()
    res['std_controller_right_vel_x'] = s_controller_right_vel_x.std()
    res['std_controller_right_vel_y'] = s_controller_right_vel_y.std()
    res['std_controller_right_vel_z'] = s_controller_right_vel_z.std()
    res['std_controller_right_angularVel_x'] = s_controller_right_angularVel_x.std()
    res['std_controller_right_angularVel_y'] = s_controller_right_angularVel_y.std()
    res['std_controller_right_angularVel_z'] = s_controller_right_angularVel_z.std()
    res['std_controller_right_pos_x'] = s_controller_right_pos_x.std()
    res['std_controller_right_pos_y'] = s_controller_right_pos_y.std()
    res['std_controller_right_pos_z'] = s_controller_right_pos_z.std()
    res['std_controller_right_rot_x'] = s_controller_right_rot_x.std()
    res['std_controller_right_rot_y'] = s_controller_right_rot_y.std()
    res['std_controller_right_rot_z'] = s_controller_right_rot_z.std()

    # user
    res['user'] = file_name[0:3]

    return res

# print(preprocess("CAR_01.csv"))


def combine_files(dir: str, file_out: str, code: int):
    files = os.listdir(dir) 

    data_dicts = []

    for file in files:
        data_dict = preprocess(file, code)
        data_dicts.append(data_dict)

    # field names
    # fields = ['time', 'headset_vel.x', 'headset_vel.y', 'headset_vel.z', 'headset_angularVel.x', 'headset_angularVel.y', 'headset_angularVel.z', 'headset_pos.x', 'headset_pos.y', 'headset_pos.z', 'headset_rot.x', 'headset_rot.y', 'headset_rot.z', 'controller_left_vel.x', 'controller_left_vel.y', 'controller_left_vel.z', 'controller_left_angularVel.x', 'controller_left_angularVel.y', 'controller_left_angularVel.z', 'controller_left_pos.x', 'controller_left_pos.y', 'controller_left_pos.z', 'controller_left_rot.x', 'controller_left_rot.y', 'controller_left_rot.z', 'controller_right_vel.x', 'controller_right_vel.y', 'controller_right_vel.z', 'controller_right_angularVel.x', 'controller_right_angularVel.y', 'controller_right_angularVel.z', 'controller_right_pos.x', 'controller_right_pos.y', 'controller_right_pos.z', 'controller_right_rot.x', 'controller_right_rot.y', 'controller_right_rot.z']
    # fields = ['peaks_controller_left_rot_x', 'std_controller_left_rot_y', 'std_controller_right_rot_x', 'mean_controller_left_pos_z', 'mean_controller_right_rot_x', 'std_controller_right_vel_y', 'mean_headset_pos_x', 'mean_controller_right_angularVel_y', 'std_controller_right_rot_z', 'mean_headset_angularVel_x', 'mean_controller_left_angularVel_y', 'peaks_controller_left_angularVel_z', 'mean_headset_pos_z', 'std_headset_rot_z', 'peaks_controller_left_vel_x', 'peaks_controller_left_vel_z', 'peaks_controller_right_angularVel_y', 'std_controller_right_pos_x', 'std_controller_left_pos_y', 'peaks_controller_right_pos_y', 'peaks_headset_vel_y', 'mean_controller_right_vel_x', 'peaks_controller_left_pos_z', 'mean_controller_right_pos_z', 'std_headset_vel_z', 'peaks_headset_angularVel_x', 'mean_headset_pos_y', 'mean_headset_rot_z', 'peaks_controller_left_angularVel_x', 'user', 'mean_controller_left_rot_z', 'std_headset_pos_y', 'std_controller_right_vel_z', 'peaks_controller_left_vel_y', 'mean_controller_right_vel_y', 'std_headset_angularVel_y', 'std_controller_left_vel_x', 'mean_controller_left_vel_z', 'std_controller_left_angularVel_z', 'mean_controller_left_pos_x', 'std_controller_right_angularVel_y', 'mean_controller_right_angularVel_x', 'peaks_controller_right_vel_y', 'std_headset_pos_z', 'peaks_headset_angularVel_y', 'std_controller_right_angularVel_x', 'peaks_controller_left_rot_z', 'mean_headset_vel_y', 'std_controller_left_vel_y', 'std_controller_left_pos_z', 'std_controller_left_rot_z', 'peaks_headset_vel_x', 'peaks_controller_right_pos_x', 'std_headset_angularVel_x', 'mean_controller_left_rot_x', 'mean_controller_left_angularVel_z', 'peaks_controller_right_rot_x', 'mean_controller_left_vel_y', 'mean_controller_right_rot_y', 'std_controller_left_angularVel_x', 'mean_controller_left_rot_y', 'std_controller_right_rot_y', 'mean_controller_right_angularVel_z', 'std_headset_rot_x', 'std_headset_rot_y', 'peaks_controller_left_pos_y', 'std_controller_left_rot_x', 'std_controller_right_pos_z', 'peaks_controller_right_vel_x', 'peaks_controller_left_angularVel_y', 'mean_headset_angularVel_y', 'mean_controller_right_pos_y', 'std_controller_left_pos_x', 'peaks_controller_right_rot_z', 'peaks_controller_left_rot_y', 'peaks_controller_left_pos_x', 'mean_headset_rot_x', 'peaks_controller_right_angularVel_z', 'std_headset_pos_x', 'mean_controller_right_rot_z', 'peaks_controller_right_vel_z', 'mean_headset_vel_x', 'mean_controller_right_pos_x', 'std_controller_right_vel_x', 'mean_controller_left_angularVel_x', 'std_controller_left_angularVel_y', 'peaks_controller_right_angularVel_x', 'mean_headset_vel_z', 'mean_headset_rot_y', 'std_headset_angularVel_z', 'std_controller_right_pos_y', 'std_headset_vel_y', 'mean_controller_right_vel_z', 'peaks_headset_angularVel_z', 'peaks_controller_right_pos_z', 'std_controller_right_angularVel_z', 'mean_controller_left_pos_y', 'mean_controller_left_vel_x', 'std_headset_vel_x', 'std_controller_left_vel_z', 'peaks_controller_right_rot_y', 'peaks_headset_vel_z', 'mean_headset_angularVel_z']

    fields = ['peaks_headset_vel_x', 'peaks_headset_vel_y', 'peaks_headset_vel_z', 'peaks_headset_angularVel_x', 'peaks_headset_angularVel_y', 'peaks_headset_angularVel_z', 'peaks_headset_rot_x', 'peaks_headset_rot_y', 'peaks_headset_rot_z', 'peaks_controller_left_vel_x', 'peaks_controller_left_vel_y', 'peaks_controller_left_vel_z', 'peaks_controller_left_angularVel_x', 'peaks_controller_left_angularVel_y', 'peaks_controller_left_angularVel_z', 'peaks_controller_left_pos_x', 'peaks_controller_left_pos_y', 'peaks_controller_left_pos_z', 'peaks_controller_left_rot_x', 'peaks_controller_left_rot_y', 'peaks_controller_left_rot_z', 'peaks_controller_right_vel_x', 'peaks_controller_right_vel_y', 'peaks_controller_right_vel_z', 'peaks_controller_right_angularVel_x', 'peaks_controller_right_angularVel_y', 'peaks_controller_right_angularVel_z', 'peaks_controller_right_pos_x', 'peaks_controller_right_pos_y', 'peaks_controller_right_pos_z', 'peaks_controller_right_rot_x', 'peaks_controller_right_rot_y', 'peaks_controller_right_rot_z', 'mean_headset_vel_x', 'mean_headset_vel_y', 'mean_headset_vel_z', 'mean_headset_angularVel_x', 'mean_headset_angularVel_y', 'mean_headset_angularVel_z', 'mean_headset_pos_x', 'mean_headset_pos_y', 'mean_headset_pos_z', 'mean_headset_rot_x', 'mean_headset_rot_y', 'mean_headset_rot_z', 'mean_controller_left_vel_x', 'mean_controller_left_vel_y', 'mean_controller_left_vel_z', 'mean_controller_left_angularVel_x', 'mean_controller_left_angularVel_y', 'mean_controller_left_angularVel_z', 'mean_controller_left_pos_x', 'mean_controller_left_pos_y', 'mean_controller_left_pos_z', 'mean_controller_left_rot_x', 'mean_controller_left_rot_y', 'mean_controller_left_rot_z', 'mean_controller_right_vel_x', 'mean_controller_right_vel_y', 'mean_controller_right_vel_z', 'mean_controller_right_angularVel_x', 'mean_controller_right_angularVel_y', 'mean_controller_right_angularVel_z', 'mean_controller_right_pos_x', 'mean_controller_right_pos_y', 'mean_controller_right_pos_z', 'mean_controller_right_rot_x', 'mean_controller_right_rot_y', 'mean_controller_right_rot_z', 'std_headset_vel_x', 'std_headset_vel_y', 'std_headset_vel_z', 'std_headset_angularVel_x', 'std_headset_angularVel_y', 'std_headset_angularVel_z', 'std_headset_pos_x', 'std_headset_pos_y', 'std_headset_pos_z', 'std_headset_rot_x', 'std_headset_rot_y', 'std_headset_rot_z', 'std_controller_left_vel_x', 'std_controller_left_vel_y', 'std_controller_left_vel_z', 'std_controller_left_angularVel_x', 'std_controller_left_angularVel_y', 'std_controller_left_angularVel_z', 'std_controller_left_pos_x', 'std_controller_left_pos_y', 'std_controller_left_pos_z', 'std_controller_left_rot_x', 'std_controller_left_rot_y', 'std_controller_left_rot_z', 'std_controller_right_vel_x', 'std_controller_right_vel_y', 'std_controller_right_vel_z', 'std_controller_right_angularVel_x', 'std_controller_right_angularVel_y', 'std_controller_right_angularVel_z', 'std_controller_right_pos_x', 'std_controller_right_pos_y', 'std_controller_right_pos_z', 'std_controller_right_rot_x', 'std_controller_right_rot_y', 'std_controller_right_rot_z', 'user']

    with open(file_out, "w") as csvfile:
        # create a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # write header and data rows
        writer.writeheader()
        writer.writerows(data_dicts)

combine_files("Data/Lab3/Labeled", "0-combined_data.csv", 0)
combine_files("Data/Lab3/Adversary", "0-adversary.csv", 1)

def create_dtree(combined_data: str):
    d = {'CAR': 0, 'QUI': 1, "URU": 2}
    df = pandas.read_csv(combined_data)

    df['user'] = df['user'].map(d)

    features = ['peaks_headset_vel_x', 'peaks_headset_vel_y', 'peaks_headset_vel_z', 'peaks_headset_angularVel_x', 'peaks_headset_angularVel_y', 'peaks_headset_angularVel_z', 'peaks_headset_rot_x', 'peaks_headset_rot_y', 'peaks_headset_rot_z', 'peaks_controller_left_vel_x', 'peaks_controller_left_vel_y', 'peaks_controller_left_vel_z', 'peaks_controller_left_angularVel_x', 'peaks_controller_left_angularVel_y', 'peaks_controller_left_angularVel_z', 'peaks_controller_left_pos_x', 'peaks_controller_left_pos_y', 'peaks_controller_left_pos_z', 'peaks_controller_left_rot_x', 'peaks_controller_left_rot_y', 'peaks_controller_left_rot_z', 'peaks_controller_right_vel_x', 'peaks_controller_right_vel_y', 'peaks_controller_right_vel_z', 'peaks_controller_right_angularVel_x', 'peaks_controller_right_angularVel_y', 'peaks_controller_right_angularVel_z', 'peaks_controller_right_pos_x', 'peaks_controller_right_pos_y', 'peaks_controller_right_pos_z', 'peaks_controller_right_rot_x', 'peaks_controller_right_rot_y', 'peaks_controller_right_rot_z', 'mean_headset_vel_x', 'mean_headset_vel_y', 'mean_headset_vel_z', 'mean_headset_angularVel_x', 'mean_headset_angularVel_y', 'mean_headset_angularVel_z', 'mean_headset_pos_x', 'mean_headset_pos_y', 'mean_headset_pos_z', 'mean_headset_rot_x', 'mean_headset_rot_y', 'mean_headset_rot_z', 'mean_controller_left_vel_x', 'mean_controller_left_vel_y', 'mean_controller_left_vel_z', 'mean_controller_left_angularVel_x', 'mean_controller_left_angularVel_y', 'mean_controller_left_angularVel_z', 'mean_controller_left_pos_x', 'mean_controller_left_pos_y', 'mean_controller_left_pos_z', 'mean_controller_left_rot_x', 'mean_controller_left_rot_y', 'mean_controller_left_rot_z', 'mean_controller_right_vel_x', 'mean_controller_right_vel_y', 'mean_controller_right_vel_z', 'mean_controller_right_angularVel_x', 'mean_controller_right_angularVel_y', 'mean_controller_right_angularVel_z', 'mean_controller_right_pos_x', 'mean_controller_right_pos_y', 'mean_controller_right_pos_z', 'mean_controller_right_rot_x', 'mean_controller_right_rot_y', 'mean_controller_right_rot_z', 'std_headset_vel_x', 'std_headset_vel_y', 'std_headset_vel_z', 'std_headset_angularVel_x', 'std_headset_angularVel_y', 'std_headset_angularVel_z', 'std_headset_pos_x', 'std_headset_pos_y', 'std_headset_pos_z', 'std_headset_rot_x', 'std_headset_rot_y', 'std_headset_rot_z', 'std_controller_left_vel_x', 'std_controller_left_vel_y', 'std_controller_left_vel_z', 'std_controller_left_angularVel_x', 'std_controller_left_angularVel_y', 'std_controller_left_angularVel_z', 'std_controller_left_pos_x', 'std_controller_left_pos_y', 'std_controller_left_pos_z', 'std_controller_left_rot_x', 'std_controller_left_rot_y', 'std_controller_left_rot_z', 'std_controller_right_vel_x', 'std_controller_right_vel_y', 'std_controller_right_vel_z', 'std_controller_right_angularVel_x', 'std_controller_right_angularVel_y', 'std_controller_right_angularVel_z', 'std_controller_right_pos_x', 'std_controller_right_pos_y', 'std_controller_right_pos_z', 'std_controller_right_rot_x', 'std_controller_right_rot_y', 'std_controller_right_rot_z']

    X = df[features].values
    y = df['user']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = DecisionTreeClassifier(criterion='gini', max_depth=1, min_samples_leaf=10, min_samples_split=20)
    clf.fit(X_train, y_train)

    # Predict on the training set
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Predict on the testing set
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Compare the accuracy on the training set and the testing set
    print("Accuracy on the training set:", train_accuracy)
    print("Accuracy on the testing set:", test_accuracy)

    print(clf.predict_proba(X_test))
    print(classification_report(y_test, y_test_pred))

    # pre-pruning 
    """
    # define the parameter grid
    param_grid = {'criterion':["gini","entropy"],
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf': [10, 20, 30],
    'min_samples_split': [20, 30, 40]}

    # Create an instance of the GridSearchCV
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("Best hyperparameters: ", grid_search.best_params_)
    # Best hyperparameters:  {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 10, 'min_samples_split': 20}
    """

# create_dtree("0-combined_data.csv")

def with_adversary(combined_data: str, adv_combined_data: str):
    d = {'CAR': 0, 'QUI': 1, "URU": 2}
    df = pandas.read_csv(combined_data)

    df_adv = pandas.read_csv(adv_combined_data)

    df['user'] = df['user'].map(d)
    df_adv['user'] = df_adv['user'].map(d)

    features = ['peaks_headset_vel_x', 'peaks_headset_vel_y', 'peaks_headset_vel_z', 'peaks_headset_angularVel_x', 'peaks_headset_angularVel_y', 'peaks_headset_angularVel_z', 'peaks_headset_rot_x', 'peaks_headset_rot_y', 'peaks_headset_rot_z', 'peaks_controller_left_vel_x', 'peaks_controller_left_vel_y', 'peaks_controller_left_vel_z', 'peaks_controller_left_angularVel_x', 'peaks_controller_left_angularVel_y', 'peaks_controller_left_angularVel_z', 'peaks_controller_left_pos_x', 'peaks_controller_left_pos_y', 'peaks_controller_left_pos_z', 'peaks_controller_left_rot_x', 'peaks_controller_left_rot_y', 'peaks_controller_left_rot_z', 'peaks_controller_right_vel_x', 'peaks_controller_right_vel_y', 'peaks_controller_right_vel_z', 'peaks_controller_right_angularVel_x', 'peaks_controller_right_angularVel_y', 'peaks_controller_right_angularVel_z', 'peaks_controller_right_pos_x', 'peaks_controller_right_pos_y', 'peaks_controller_right_pos_z', 'peaks_controller_right_rot_x', 'peaks_controller_right_rot_y', 'peaks_controller_right_rot_z', 'mean_headset_vel_x', 'mean_headset_vel_y', 'mean_headset_vel_z', 'mean_headset_angularVel_x', 'mean_headset_angularVel_y', 'mean_headset_angularVel_z', 'mean_headset_pos_x', 'mean_headset_pos_y', 'mean_headset_pos_z', 'mean_headset_rot_x', 'mean_headset_rot_y', 'mean_headset_rot_z', 'mean_controller_left_vel_x', 'mean_controller_left_vel_y', 'mean_controller_left_vel_z', 'mean_controller_left_angularVel_x', 'mean_controller_left_angularVel_y', 'mean_controller_left_angularVel_z', 'mean_controller_left_pos_x', 'mean_controller_left_pos_y', 'mean_controller_left_pos_z', 'mean_controller_left_rot_x', 'mean_controller_left_rot_y', 'mean_controller_left_rot_z', 'mean_controller_right_vel_x', 'mean_controller_right_vel_y', 'mean_controller_right_vel_z', 'mean_controller_right_angularVel_x', 'mean_controller_right_angularVel_y', 'mean_controller_right_angularVel_z', 'mean_controller_right_pos_x', 'mean_controller_right_pos_y', 'mean_controller_right_pos_z', 'mean_controller_right_rot_x', 'mean_controller_right_rot_y', 'mean_controller_right_rot_z', 'std_headset_vel_x', 'std_headset_vel_y', 'std_headset_vel_z', 'std_headset_angularVel_x', 'std_headset_angularVel_y', 'std_headset_angularVel_z', 'std_headset_pos_x', 'std_headset_pos_y', 'std_headset_pos_z', 'std_headset_rot_x', 'std_headset_rot_y', 'std_headset_rot_z', 'std_controller_left_vel_x', 'std_controller_left_vel_y', 'std_controller_left_vel_z', 'std_controller_left_angularVel_x', 'std_controller_left_angularVel_y', 'std_controller_left_angularVel_z', 'std_controller_left_pos_x', 'std_controller_left_pos_y', 'std_controller_left_pos_z', 'std_controller_left_rot_x', 'std_controller_left_rot_y', 'std_controller_left_rot_z', 'std_controller_right_vel_x', 'std_controller_right_vel_y', 'std_controller_right_vel_z', 'std_controller_right_angularVel_x', 'std_controller_right_angularVel_y', 'std_controller_right_angularVel_z', 'std_controller_right_pos_x', 'std_controller_right_pos_y', 'std_controller_right_pos_z', 'std_controller_right_rot_x', 'std_controller_right_rot_y', 'std_controller_right_rot_z']

    X = df[features].values
    y = df['user']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_leaf=10, min_samples_split=20)
    clf.fit(X_train, y_train)

    # Predict on the adversarial set
    X_adv = df_adv[features].values
    print(clf.predict(X_adv))
    print(clf.predict_proba(X_adv))

with_adversary("0-combined_data.csv", "0-adversary.csv")