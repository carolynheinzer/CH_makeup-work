import argparse
from glob import glob
import os
import pandas
import scipy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import csv
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import sys
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def extract(file_name: str, test_train: int):
    # returns dict of all atts and their values
    if test_train == 0:
        name = "Data/Lab3/Train/" + file_name
    elif test_train == 1:
        name = file_name
    elif test_train == 2:
        name = "Data/Lab3/Validation/" + file_name
    f = open(name, "r")

    # skip first line
    line1 = f.readline()

    # want to use more data than previously to try to capture the idiosynchracies of the users
    # initialize dict of time,headset_vel.x,headset_vel.y,headset_vel.z,headset_angularVel.x,headset_angularVel.y,headset_angularVel.z,headset_pos.x,headset_pos.y,headset_pos.z,headset_rot.x,headset_rot.y,headset_rot.z,controller_left_vel.x,controller_left_vel.y,controller_left_vel.z,controller_left_angularVel.x,controller_left_angularVel.y,controller_left_angularVel.z,controller_left_pos.x,controller_left_pos.y,controller_left_pos.z,controller_left_rot.x,controller_left_rot.y,controller_left_rot.z,controller_right_vel.x,controller_right_vel.y,controller_right_vel.z,controller_right_angularVel.x,controller_right_angularVel.y,controller_right_angularVel.z,controller_right_pos.x,controller_right_pos.y,controller_right_pos.z,controller_right_rot.x,controller_right_rot.y,controller_right_rot.z
    info = {}
    info['time'] = []
    info['headset_vel_x'] = []
    info['headset_vel_y'] = []
    info['headset_vel_z'] = []
    info['headset_angularVel_x'] = []
    info['headset_angularVel_y'] = []
    info['headset_angularVel_z'] = []
    info['headset_pos_x'] = []
    info['headset_pos_y'] = []
    info['headset_pos_z'] = []
    info['headset_rot_x'] = []
    info['headset_rot_y'] = []
    info['headset_rot_z'] = []
    info['controller_left_vel_x'] = []
    info['controller_left_vel_y'] = []
    info['controller_left_vel_z'] = []
    info['controller_left_angularVel_x'] = []
    info['controller_left_angularVel_y'] = []
    info['controller_left_angularVel_z'] = []
    info['controller_left_pos_x'] = []
    info['controller_left_pos_y'] = []
    info['controller_left_pos_z'] = []
    info['controller_left_rot_x'] = []
    info['controller_left_rot_y'] = []
    info['controller_left_rot_z'] = []
    info['controller_right_vel_x'] = []
    info['controller_right_vel_y'] = []
    info['controller_right_vel_z'] = []
    info['controller_right_angularVel_x'] = []
    info['controller_right_angularVel_y'] = []
    info['controller_right_angularVel_z'] = []
    info['controller_right_pos_x'] = []
    info['controller_right_pos_y'] = []
    info['controller_right_pos_z'] = []
    info['controller_right_rot_x'] = []
    info['controller_right_rot_y'] = []
    info['controller_right_rot_z'] = []

    # put appropriate data into dictionary

    atts = list(info.keys())
    for line in f.readlines():
        att_vals = line.split(",")
        
        for i in range(0, 36):
            info[atts[i]].append(att_vals[i])

    f.close()
    return info

# print(extract("CAR_02.csv", 0))

def preprocess(file_name: str, test_train: int):
    # finds the mean, standard deviation, and number of peaks
    # returns dict containing all relevant column values for dataframe
    info = extract(file_name, test_train)

    # initialize results variable
    res = {}

    # x values - time
    times = list(map(float, info['time']))

    # y values - the rest of the attributes
    v_headset_vel_x = list(map(float, info['headset_vel_x']))
    v_headset_vel_y = list(map(float, info['headset_vel_y']))
    v_headset_vel_z = list(map(float, info['headset_vel_z']))
    v_headset_angularVel_x = list(map(float, info['headset_angularVel_x']))
    v_headset_angularVel_y = list(map(float, info['headset_angularVel_y']))
    v_headset_angularVel_z = list(map(float, info['headset_angularVel_z']))
    v_headset_pos_x = list(map(float, info['headset_pos_x']))
    v_headset_pos_y = list(map(float, info['headset_pos_y']))
    v_headset_pos_z = list(map(float, info['headset_pos_z']))
    v_headset_rot_x = list(map(float, info['headset_rot_x']))
    v_headset_rot_y = list(map(float, info['headset_rot_y']))
    v_headset_rot_z = list(map(float, info['headset_rot_z']))
    v_controller_left_vel_x = list(map(float, info['controller_left_vel_x']))
    v_controller_left_vel_y = list(map(float, info['controller_left_vel_y']))
    v_controller_left_vel_z = list(map(float, info['controller_left_vel_z']))
    v_controller_left_angularVel_x = list(map(float, info['controller_left_angularVel_x']))
    v_controller_left_angularVel_y = list(map(float, info['controller_left_angularVel_y']))
    v_controller_left_angularVel_z = list(map(float, info['controller_left_angularVel_z']))
    v_controller_left_pos_x = list(map(float, info['controller_left_pos_x']))
    v_controller_left_pos_y = list(map(float, info['controller_left_pos_y']))
    v_controller_left_pos_z = list(map(float, info['controller_left_pos_z']))
    v_controller_left_rot_x = list(map(float, info['controller_left_rot_x']))
    v_controller_left_rot_y = list(map(float, info['controller_left_rot_y']))
    v_controller_left_rot_z = list(map(float, info['controller_left_rot_z']))
    v_controller_right_vel_x = list(map(float, info['controller_right_vel_x']))
    v_controller_right_vel_y = list(map(float, info['controller_right_vel_y']))
    v_controller_right_vel_z = list(map(float, info['controller_right_vel_z']))
    v_controller_right_angularVel_x = list(map(float, info['controller_right_angularVel_x']))
    v_controller_right_angularVel_y = list(map(float, info['controller_right_angularVel_y']))
    v_controller_right_angularVel_z = list(map(float, info['controller_right_angularVel_z']))
    v_controller_right_pos_x = list(map(float, info['controller_right_pos_x']))
    v_controller_right_pos_y = list(map(float, info['controller_right_pos_y']))
    v_controller_right_pos_z = list(map(float, info['controller_right_pos_z']))
    v_controller_right_rot_x = list(map(float, info['controller_right_rot_x']))
    v_controller_right_rot_y = list(map(float, info['controller_right_rot_y']))
    v_controller_right_rot_z = list(map(float, info['controller_right_rot_z']))

    # find the numbers of peaks in these sets of y values
    peaks_headset_vel_x, _ = find_peaks(v_headset_vel_x)
    peaks_headset_vel_y, _ = find_peaks(v_headset_vel_y)
    peaks_headset_vel_z, _ = find_peaks(v_headset_vel_z)
    peaks_headset_angularVel_x, _ = find_peaks(v_headset_angularVel_x)
    peaks_headset_angularVel_y, _ = find_peaks(v_headset_angularVel_y)
    peaks_headset_angularVel_z, _ = find_peaks(v_headset_angularVel_z)
    peaks_headset_pos_x, _ = find_peaks(v_headset_pos_x)
    peaks_headset_pos_y, _ = find_peaks(v_headset_pos_y)
    peaks_headset_pos_z, _ = find_peaks(v_headset_pos_z)
    peaks_headset_rot_x, _ = find_peaks(v_headset_rot_x)
    peaks_headset_rot_y, _ = find_peaks(v_headset_rot_y)
    peaks_headset_rot_z, _ = find_peaks(v_headset_rot_z)
    peaks_controller_left_vel_x, _ = find_peaks(v_controller_left_vel_x)
    peaks_controller_left_vel_y, _ = find_peaks(v_controller_left_vel_y)
    peaks_controller_left_vel_z, _ = find_peaks(v_controller_left_vel_z)
    peaks_controller_left_angularVel_x, _ = find_peaks(v_controller_left_angularVel_x)
    peaks_controller_left_angularVel_y, _ = find_peaks(v_controller_left_angularVel_y)
    peaks_controller_left_angularVel_z, _ = find_peaks(v_controller_left_angularVel_z)
    peaks_controller_left_pos_x, _ = find_peaks(v_controller_left_pos_x)
    peaks_controller_left_pos_y, _ = find_peaks(v_controller_left_pos_y)
    peaks_controller_left_pos_z, _ = find_peaks(v_controller_left_pos_z)
    peaks_controller_left_rot_x, _ = find_peaks(v_controller_left_rot_x)
    peaks_controller_left_rot_y, _ = find_peaks(v_controller_left_rot_y)
    peaks_controller_left_rot_z, _ = find_peaks(v_controller_left_rot_z)
    peaks_controller_right_vel_x, _ = find_peaks(v_controller_right_vel_x)
    peaks_controller_right_vel_y, _ = find_peaks(v_controller_right_vel_y)
    peaks_controller_right_vel_z, _ = find_peaks(v_controller_right_vel_z)
    peaks_controller_right_angularVel_x, _ = find_peaks(v_controller_right_angularVel_x)
    peaks_controller_right_angularVel_y, _ = find_peaks(v_controller_right_angularVel_y)
    peaks_controller_right_angularVel_z, _ = find_peaks(v_controller_right_angularVel_z)
    peaks_controller_right_pos_x, _ = find_peaks(v_controller_right_pos_x)
    peaks_controller_right_pos_y, _ = find_peaks(v_controller_right_pos_y)
    peaks_controller_right_pos_z, _ = find_peaks(v_controller_right_pos_z)
    peaks_controller_right_rot_x, _ = find_peaks(v_controller_right_rot_x)
    peaks_controller_right_rot_y, _ = find_peaks(v_controller_right_rot_y)
    # peaks_controller_right_rot_z, _ = find_peaks(v_controller_right_rot_z)

    # populate the results dictionary with the number of peaks
    res['peaks_headset_vel_x'] = len(peaks_headset_vel_x)
    res['peaks_headset_vel_y'] = len(peaks_headset_vel_y)
    res['peaks_headset_vel_z'] = len(peaks_headset_vel_z)
    res['peaks_headset_angularVel_x'] = len(peaks_headset_angularVel_x)
    res['peaks_headset_angularVel_y'] = len(peaks_headset_angularVel_y)
    res['peaks_headset_angularVel_z'] = len(peaks_headset_angularVel_z)
    res['peaks_headset_pos_x'] = len(peaks_headset_pos_x)
    res['peaks_headset_pos_y'] = len(peaks_headset_pos_y)
    res['peaks_headset_pos_z'] = len(peaks_headset_pos_z)
    res['peaks_headset_rot_x'] = len(peaks_headset_rot_x)
    res['peaks_headset_rot_y'] = len(peaks_headset_rot_y)
    res['peaks_headset_rot_z'] = len(peaks_headset_rot_z)
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
    # res['peaks_controller_right_rot_z'] = len(peaks_controller_right_rot_z)

    # find mean of all attributes

    # create the series
    s_headset_vel_x = pandas.Series(data = v_headset_vel_x)
    s_headset_vel_y = pandas.Series(data = v_headset_vel_y)
    s_headset_vel_z = pandas.Series(data = v_headset_vel_z)
    s_headset_angularVel_x = pandas.Series(data = v_headset_angularVel_x)
    s_headset_angularVel_y = pandas.Series(data = v_headset_angularVel_y)
    s_headset_angularVel_z = pandas.Series(data = v_headset_angularVel_z)
    s_headset_pos_x = pandas.Series(data = v_headset_pos_x)
    s_headset_pos_y = pandas.Series(data = v_headset_pos_y)
    s_headset_pos_z = pandas.Series(data = v_headset_pos_z)
    s_headset_rot_x = pandas.Series(data = v_headset_rot_x)
    s_headset_rot_y = pandas.Series(data = v_headset_rot_y)
    s_headset_rot_z = pandas.Series(data = v_headset_rot_z)
    s_controller_left_vel_x = pandas.Series(data = v_controller_left_vel_x)
    s_controller_left_vel_y = pandas.Series(data = v_controller_left_vel_y)
    s_controller_left_vel_z = pandas.Series(data = v_controller_left_vel_z)
    s_controller_left_angularVel_x = pandas.Series(data = v_controller_left_angularVel_x)
    s_controller_left_angularVel_y = pandas.Series(data = v_controller_left_angularVel_y)
    s_controller_left_angularVel_z = pandas.Series(data = v_controller_left_angularVel_z)
    s_controller_left_pos_x = pandas.Series(data = v_controller_left_pos_x)
    s_controller_left_pos_y = pandas.Series(data = v_controller_left_pos_y)
    s_controller_left_pos_z = pandas.Series(data = v_controller_left_pos_z)
    s_controller_left_rot_x = pandas.Series(data = v_controller_left_rot_x)
    s_controller_left_rot_y = pandas.Series(data = v_controller_left_rot_y)
    s_controller_left_rot_z = pandas.Series(data = v_controller_left_rot_z)
    s_controller_right_vel_x = pandas.Series(data = v_controller_right_vel_x)
    s_controller_right_vel_y = pandas.Series(data = v_controller_right_vel_y)
    s_controller_right_vel_z = pandas.Series(data = v_controller_right_vel_z)
    s_controller_right_angularVel_x = pandas.Series(data = v_controller_right_angularVel_x)
    s_controller_right_angularVel_y = pandas.Series(data = v_controller_right_angularVel_y)
    s_controller_right_angularVel_z = pandas.Series(data = v_controller_right_angularVel_z)
    s_controller_right_pos_x = pandas.Series(data = v_controller_right_pos_x)
    s_controller_right_pos_y = pandas.Series(data = v_controller_right_pos_y)
    s_controller_right_pos_z = pandas.Series(data = v_controller_right_pos_z)
    s_controller_right_rot_x = pandas.Series(data = v_controller_right_rot_x)
    s_controller_right_rot_y = pandas.Series(data = v_controller_right_rot_y)
    s_controller_right_rot_z = pandas.Series(data = v_controller_right_rot_z)

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
    #res['mean_controller_right_rot_z'] = s_controller_right_rot_z.mean()

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
    # res['std_controller_right_rot_z'] = s_controller_right_rot_z.std()

    # user
    res['user'] = file_name[0:3]

    return res

# print(preprocess("CAR_02.csv", 0))

def combine_files(dir: str, file_out: str, code: int):
    # create a dataframe from all files in Train

    # files = os.listdir(dir)
    data_dicts = []

    for file in dir:
        data_dict = preprocess(file, code)
        data_dicts.append(data_dict)

    # field names - cut peaks, mean, and std_controller_right_z because was nan???
    fields = ['peaks_headset_vel_x', 'peaks_headset_vel_y', 'peaks_headset_vel_z', 'peaks_headset_angularVel_x', 'peaks_headset_angularVel_y', 'peaks_headset_angularVel_z', 'peaks_headset_pos_x', 'peaks_headset_pos_y', 'peaks_headset_pos_z', 'peaks_headset_rot_x', 'peaks_headset_rot_y', 'peaks_headset_rot_z', 'peaks_controller_left_vel_x', 'peaks_controller_left_vel_y', 'peaks_controller_left_vel_z', 'peaks_controller_left_angularVel_x', 'peaks_controller_left_angularVel_y', 'peaks_controller_left_angularVel_z', 'peaks_controller_left_pos_x', 'peaks_controller_left_pos_y', 'peaks_controller_left_pos_z', 'peaks_controller_left_rot_x', 'peaks_controller_left_rot_y', 'peaks_controller_left_rot_z', 'peaks_controller_right_vel_x', 'peaks_controller_right_vel_y', 'peaks_controller_right_vel_z', 'peaks_controller_right_angularVel_x', 'peaks_controller_right_angularVel_y', 'peaks_controller_right_angularVel_z', 'peaks_controller_right_pos_x', 'peaks_controller_right_pos_y', 'peaks_controller_right_pos_z', 'peaks_controller_right_rot_x', 'peaks_controller_right_rot_y', 'mean_headset_vel_x', 'mean_headset_vel_y', 'mean_headset_vel_z', 'mean_headset_angularVel_x', 'mean_headset_angularVel_y', 'mean_headset_angularVel_z', 'mean_headset_pos_x', 'mean_headset_pos_y', 'mean_headset_pos_z', 'mean_headset_rot_x', 'mean_headset_rot_y', 'mean_headset_rot_z', 'mean_controller_left_vel_x', 'mean_controller_left_vel_y', 'mean_controller_left_vel_z', 'mean_controller_left_angularVel_x', 'mean_controller_left_angularVel_y', 'mean_controller_left_angularVel_z', 'mean_controller_left_pos_x', 'mean_controller_left_pos_y', 'mean_controller_left_pos_z', 'mean_controller_left_rot_x', 'mean_controller_left_rot_y', 'mean_controller_left_rot_z', 'mean_controller_right_vel_x', 'mean_controller_right_vel_y', 'mean_controller_right_vel_z', 'mean_controller_right_angularVel_x', 'mean_controller_right_angularVel_y', 'mean_controller_right_angularVel_z', 'mean_controller_right_pos_x', 'mean_controller_right_pos_y', 'mean_controller_right_pos_z', 'mean_controller_right_rot_x', 'mean_controller_right_rot_y', 'std_headset_vel_x', 'std_headset_vel_y', 'std_headset_vel_z', 'std_headset_angularVel_x', 'std_headset_angularVel_y', 'std_headset_angularVel_z', 'std_headset_pos_x', 'std_headset_pos_y', 'std_headset_pos_z', 'std_headset_rot_x', 'std_headset_rot_y', 'std_headset_rot_z', 'std_controller_left_vel_x', 'std_controller_left_vel_y', 'std_controller_left_vel_z', 'std_controller_left_angularVel_x', 'std_controller_left_angularVel_y', 'std_controller_left_angularVel_z', 'std_controller_left_pos_x', 'std_controller_left_pos_y', 'std_controller_left_pos_z', 'std_controller_left_rot_x', 'std_controller_left_rot_y', 'std_controller_left_rot_z', 'std_controller_right_vel_x', 'std_controller_right_vel_y', 'std_controller_right_vel_z', 'std_controller_right_angularVel_x', 'std_controller_right_angularVel_y', 'std_controller_right_angularVel_z', 'std_controller_right_pos_x', 'std_controller_right_pos_y', 'std_controller_right_pos_z', 'std_controller_right_rot_x', 'std_controller_right_rot_y', 'user']

    # name of csv
    filename = file_out

    # write to csv
    with open(filename, 'w') as csvfile:
        # create a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # write header and data rows
        writer.writeheader()
        writer.writerows(data_dicts)

# combine_files("Data/Lab3/Train", "combined.csv", 0)

def create_dtree(file_name: str):
    # create dataframe and ensure all data is numbers
    d = {'CAR': 0, 'QUI': 1, "URU": 2}
    df = pandas.read_csv(file_name)

    df['user'] = df['user'].map(d)

    # separate feature columns from target column
    features = ['peaks_headset_vel_x', 'peaks_headset_vel_y', 'peaks_headset_vel_z', 'peaks_headset_angularVel_x', 'peaks_headset_angularVel_y', 'peaks_headset_angularVel_z', 'peaks_headset_pos_x', 'peaks_headset_pos_y', 'peaks_headset_pos_z', 'peaks_headset_rot_x', 'peaks_headset_rot_y', 'peaks_headset_rot_z', 'peaks_controller_left_vel_x', 'peaks_controller_left_vel_y', 'peaks_controller_left_vel_z', 'peaks_controller_left_angularVel_x', 'peaks_controller_left_angularVel_y', 'peaks_controller_left_angularVel_z', 'peaks_controller_left_pos_x', 'peaks_controller_left_pos_y', 'peaks_controller_left_pos_z', 'peaks_controller_left_rot_x', 'peaks_controller_left_rot_y', 'peaks_controller_left_rot_z', 'peaks_controller_right_vel_x', 'peaks_controller_right_vel_y', 'peaks_controller_right_vel_z', 'peaks_controller_right_angularVel_x', 'peaks_controller_right_angularVel_y', 'peaks_controller_right_angularVel_z', 'peaks_controller_right_pos_x', 'peaks_controller_right_pos_y', 'peaks_controller_right_pos_z', 'peaks_controller_right_rot_x', 'peaks_controller_right_rot_y', 'mean_headset_vel_x', 'mean_headset_vel_y', 'mean_headset_vel_z', 'mean_headset_angularVel_x', 'mean_headset_angularVel_y', 'mean_headset_angularVel_z', 'mean_headset_pos_x', 'mean_headset_pos_y', 'mean_headset_pos_z', 'mean_headset_rot_x', 'mean_headset_rot_y', 'mean_headset_rot_z', 'mean_controller_left_vel_x', 'mean_controller_left_vel_y', 'mean_controller_left_vel_z', 'mean_controller_left_angularVel_x', 'mean_controller_left_angularVel_y', 'mean_controller_left_angularVel_z', 'mean_controller_left_pos_x', 'mean_controller_left_pos_y', 'mean_controller_left_pos_z', 'mean_controller_left_rot_x', 'mean_controller_left_rot_y', 'mean_controller_left_rot_z', 'mean_controller_right_vel_x', 'mean_controller_right_vel_y', 'mean_controller_right_vel_z', 'mean_controller_right_angularVel_x', 'mean_controller_right_angularVel_y', 'mean_controller_right_angularVel_z', 'mean_controller_right_pos_x', 'mean_controller_right_pos_y', 'mean_controller_right_pos_z', 'mean_controller_right_rot_x', 'mean_controller_right_rot_y', 'std_headset_vel_x', 'std_headset_vel_y', 'std_headset_vel_z', 'std_headset_angularVel_x', 'std_headset_angularVel_y', 'std_headset_angularVel_z', 'std_headset_pos_x', 'std_headset_pos_y', 'std_headset_pos_z', 'std_headset_rot_x', 'std_headset_rot_y', 'std_headset_rot_z', 'std_controller_left_vel_x', 'std_controller_left_vel_y', 'std_controller_left_vel_z', 'std_controller_left_angularVel_x', 'std_controller_left_angularVel_y', 'std_controller_left_angularVel_z', 'std_controller_left_pos_x', 'std_controller_left_pos_y', 'std_controller_left_pos_z', 'std_controller_left_rot_x', 'std_controller_left_rot_y', 'std_controller_left_rot_z', 'std_controller_right_vel_x', 'std_controller_right_vel_y', 'std_controller_right_vel_z', 'std_controller_right_angularVel_x', 'std_controller_right_angularVel_y', 'std_controller_right_angularVel_z', 'std_controller_right_pos_x', 'std_controller_right_pos_y', 'std_controller_right_pos_z', 'std_controller_right_rot_x', 'std_controller_right_rot_y']

    X = df[features].values
    y = df['user']

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)
    return dtree

# create_dtree("combined.csv")

def predict_shallow(sensor_data: str) -> str:
    # Run prediction on an sensor data sample.

    dtree = create_dtree("lab3_data_summary.csv")

    # extract the input for the prediction
    ipt = preprocess(sensor_data, 1)
    lst_ipt = []

    for key, val in ipt.items():
        if (key != 'user'):
            lst_ipt.append(val)

    user = dtree.predict([lst_ipt])     

    result = 100    # garbage value

    if (user == 0):
        result = "CAR"
    elif (user == 1):
        result = "QUI"
    elif (user == 2):
        result = "URU"

    return result

# print(predict_shallow("Data/Lab3/Validation/CAR_01.csv"))

def calc_accuracy():
    # accuracy and precision reports
    val_files = os.listdir("Data/Lab3/Validation")
    combine_files(val_files, "val_data_summary.csv", 2)

    test = []
    pred = []

    for file in val_files:
        test.append(predict_shallow("Data/Lab3/Validation/" + file))
        pred.append(file[0:3])

    # convert to ints
    test_vals = []
    pred_vals = []

    for arr in test, pred:
        for el in arr:
            if ("CAR" in el):
                test_vals.append(0)
                pred_vals.append(0)
            elif ("QUI" in el):
                test_vals.append(1)
                pred_vals.append(1)
            elif ("URU" in el):
                test_vals.append(2)
                pred_vals.append(2)
            
    target_names = ["CAR", "QUI", "URU"]
    print(classification_report(pred_vals, test_vals, target_names=target_names))

def predict_shallow_folder(data_folder: str, output: str):
    # Run the model's prediction on all the sensor data in data_folder, writing labels
    # in sequence to an output text file.

    # create data summary
    files = os.listdir("Data/Lab3/Train")
    out_file = "lab3_data_summary.csv"
    combine_files(files, out_file, 0)

    data_files = sorted(glob(f"{data_folder}/*.csv"))
    data_files = [f for f in data_files if 'data_summary' not in f]
    labels = map(predict_shallow, data_files)

    with open(output, "w+") as output_file:
        output_file.write("\n".join(labels))


if __name__ == "__main__":
    # Parse arguments to determine whether to predict on a file or a folder
    # You should not need to modify the below starter code, but feel free to
    # add more arguments for debug functions as needed.
    parser = argparse.ArgumentParser()

    sample_input = parser.add_mutually_exclusive_group(required=True)
    sample_input.add_argument(
        "sample", nargs="?", help="A .csv sensor data file to run predictions on"
    )
    sample_input.add_argument(
        "--label_folder",
        type=str,
        required=False,
        help="Folder of .csv data files to run predictions on",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="Data/Lab3/shallow-lab3.txt",
        help="Output filename of labels when running predictions on a directory",
    )

    args = parser.parse_args()

    if args.sample:
        print(predict_shallow(args.sample))

    elif args.label_folder:
        predict_shallow_folder(args.label_folder, args.output)

    calc_accuracy()