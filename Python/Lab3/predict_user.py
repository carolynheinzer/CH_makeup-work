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

    atts = info.keys()
    for line in f.readlines():
        att_vals = line.split(",")
        
        for i in range(0, 36):
            info[atts[i]].append(att_vals[i])

    print(info)

    f.close()
    return info

extract("../Data/Lab3/Carolyn_data/ARC_01.csv", 0)
