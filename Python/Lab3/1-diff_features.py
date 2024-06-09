import argparse
from glob import glob
import os
import pandas
import sys
import numpy as np

import scipy
from scipy.signal import find_peaks

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import csv

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.svm import SVC

def to_dict(file_name: str, code: int):
    # return dict of attributes and their values

    if code == 0:
        name = "Data/Lab3/Train/" + file_name
    elif code == 1:
        name = file_name
    
    f = open(name, "r")

    # get keys
    info = {}
    firstline = f.readline()

    # remove newlines
    split = firstline.split(",")
    keys = [key.replace("\n", "") for key in split]
    keys = [key.replace(".", "_") for key in keys]

    for key in keys:
        info[key] = []

    for line in f.readlines():
        split = line.split(",")
        att_vals = [val.replace("\n", "") for val in split]

        for i in range(0, 37):
            el = att_vals[i]
            info[keys[i]].append(float(el))

    return info

# to_dict("CAR_02.csv", 0)

def preprocess(file_name: str, code: int):
    # put relevant attributes into a dictionary
    info = to_dict(file_name, code)

    # results
    res = {}

    rel_features = ['headset_pos_x', 'headset_pos_y', 'headset_pos_z', 'controller_left_vel_x', 'controller_left_vel_y', 'controller_left_vel_z', 'controller_left_pos_x', 'controller_left_pos_y', 'controller_left_pos_z']

    headset_vel_z = info['headset_vel_z']
    headset_pos_z = info['headset_pos_z']

    controller_left_vel_x = info['controller_left_vel_x']
    controller_left_vel_y = info['controller_left_vel_y']
    controller_left_vel_z = info['controller_left_vel_z']
    controller_left_pos_x = info['controller_left_pos_x']
    controller_left_pos_y = info['controller_left_pos_y']
    controller_left_pos_z = info['controller_left_pos_z']

    # series
    s_headset_vel_z = pandas.Series(data = headset_vel_z)
    s_headset_pos_z = pandas.Series(data = headset_pos_z)

    s_controller_left_vel_x = pandas.Series(data = controller_left_vel_x)
    s_controller_left_vel_y = pandas.Series(data = controller_left_vel_y)
    s_controller_left_vel_z = pandas.Series(data = controller_left_vel_z)
    s_controller_left_pos_x = pandas.Series(data = controller_left_pos_x)
    s_controller_left_pos_y = pandas.Series(data = controller_left_pos_y)
    s_controller_left_pos_z = pandas.Series(data = controller_left_pos_z)

    # headset_vel_z - max
    res['max_headset_vel_z'] = s_headset_vel_z.max()

    # headset_pos_z - max
    res['max_headset_pos_z'] = s_headset_pos_z.max()

    # controller_left_vel_x,y,z - max, min
    res['max_controller_left_vel_x'] = s_controller_left_vel_x.max()
    res['max_controller_left_vel_y'] = s_controller_left_vel_y.max()
    res['max_controller_left_vel_z'] = s_controller_left_vel_z.max()
    res['min_controller_left_vel_x'] = s_controller_left_vel_x.min()
    res['min_controller_left_vel_y'] = s_controller_left_vel_y.min()
    res['min_controller_left_vel_z'] = s_controller_left_vel_z.min()

    # controller_left_pos_x,y,z - mean, max
    # for z: max - min
    res['max_controller_left_pos_x'] = s_controller_left_pos_x.max()
    res['max_controller_left_pos_y'] = s_controller_left_pos_y.max()
    res['max_controller_left_pos_z'] = s_controller_left_pos_z.max()
    res['mean_controller_left_pos_x'] = s_controller_left_pos_x.mean()
    res['mean_controller_left_pos_y'] = s_controller_left_pos_y.mean()
    res['mean_controller_left_pos_z'] = s_controller_left_pos_z.mean()
    res['min_controller_left_pos_x'] = s_controller_left_pos_x.min()
    res['min_controller_left_pos_y'] = s_controller_left_pos_y.min()
    res['min_controller_left_pos_z'] = s_controller_left_pos_z.min()
    res['diff_controller_left_pos_z'] = res['max_controller_left_pos_z'] - res['min_controller_left_pos_z']

    # user
    res['user'] = file_name[0:3]

    return res

"""
print("CAR")
preprocess("CAR_02.csv", 0)
print("QUI")
preprocess("QUI_01.csv", 0)
print("URU")
preprocess("URU_01.csv", 0)
"""

def combine_files(dir: str, file_out: str, code: int):
    # combine dictionaries of all relevant attributes from all files in a directory
    files = os.listdir(dir)

    data_dicts = []

    for file in files:
        data_dict = preprocess(file, code)
        data_dicts.append(data_dict)

    fields = ['max_headset_vel_z', 'max_headset_pos_z', 'max_controller_left_vel_x', 'max_controller_left_vel_y', 'max_controller_left_vel_z', 'min_controller_left_vel_x', 'min_controller_left_vel_y', 'min_controller_left_vel_z', 'max_controller_left_pos_x', 'max_controller_left_pos_y', 'max_controller_left_pos_z', 'mean_controller_left_pos_x', 'mean_controller_left_pos_y', 'mean_controller_left_pos_z', 'min_controller_left_pos_x', 'min_controller_left_pos_y', 'min_controller_left_pos_z', 'diff_controller_left_pos_z', 'user']

    with open(file_out, "w") as csvfile:
        # create a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # write header and data rows
        writer.writeheader()
        for data_dict in data_dicts:
            writer.writerow(data_dict)

# combine_files("Data/Lab3/Train", "1-combined.csv", 0)

def create_dtree(train_file: str):
    # create the actual shallow learning mechanism
    # ...is it even a dtree anymore??

    # create dataframe and ensure all data is numbers
    d = {'CAR': 0, 'QUI': 1, "URU": 2}
    df = pandas.read_csv(train_file)

    df['user'] = df['user'].map(d)

    # separate feature columns from target column
    features = ['max_headset_vel_z', 'max_headset_pos_z', 'max_controller_left_vel_x', 'max_controller_left_vel_y', 'max_controller_left_vel_z', 'min_controller_left_vel_x', 'min_controller_left_vel_y', 'min_controller_left_vel_z', 'max_controller_left_pos_x', 'max_controller_left_pos_y', 'max_controller_left_pos_z', 'mean_controller_left_pos_x', 'mean_controller_left_pos_y', 'mean_controller_left_pos_z', 'min_controller_left_pos_x', 'min_controller_left_pos_y', 'min_controller_left_pos_z', 'diff_controller_left_pos_z']

    X = df[features].values
    y  = df['user']

    clf = SVC(probability=True)
    clf.fit(X,y)
    # print(clf.predict_proba(X))

    # print(clf.decision_function(X))
    return clf

    """
    # not using this anymore
    dtree = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_leaf=10, min_samples_split=20)
    dtree.fit(X,y)
    return dtree"""

    """
    # GridSearchCV

    # Define the parameter grid
    param_grid = {'criterion':["gini","entropy"],
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf': [10, 20, 30],
    'min_samples_split': [20, 30, 40]}

    grid_search = GridSearchCV(dtree, param_grid, cv=5)
    grid_search.fit(X, y)
    print("Best hyperparameters: ", grid_search.best_params_)
    # Best hyperparameters:  {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 10, 'min_samples_split': 20}
    """

# create_dtree("1-combined.csv")

def predict_shallow(sensor_data: str) -> str:
    # predict the user for one data file

    clf = create_dtree("1-combined.csv")

    input = preprocess(sensor_data, 1)
    input_vals = [val for key, val in input.items() if key != 'user']
    # print(input_vals)
    # print(dtree.predict_proba([input_vals]))
    # print(clf.decision_function([input_vals]))
    # print(clf.predict_proba([input_vals]))

    user = clf.predict([input_vals])
    # user = dtree.predict([input_vals])
    result = 100    # garbage value

    if (user == 0):
        result = "CAR"
    elif (user == 1):
        result = "QUI"
    elif (user == 2):
        result = "URU"
    
    print(result)
    return result
    """
    # process with a dtree
    dtree = create_dtree("1-combined.csv")

    input = preprocess(sensor_data, 1)
    input_vals = [val for key, val in input.items() if key != 'user']
    # print(input_vals)
    print(dtree.predict_proba([input_vals]))

    user = dtree.predict([input_vals])
    result = 100    # garbage value

    if (user == 0):
        result = "CAR"
    elif (user == 1):
        result = "QUI"
    elif (user == 2):
        result = "URU"
    
    return result"""

# print(predict_shallow("CAR_01.csv"))

def predict_shallow_folder(data_folder: str, output: str):
    # Run the model's prediction on all the sensor data in data_folder, writing labels
    # in sequence to an output text file.

    data_files = sorted(glob(f"{data_folder}/*.csv"))
    labels = map(predict_shallow, data_files)
    # print(labels)

    # actual values
    actual = []
    for file in data_files:
        # print(file)
        actual.append(file[21:24])

    actual_vals = []

    for el in actual:
        if ("CAR" in el):
            actual_vals.append(0)
        elif ("QUI" in el):
            actual_vals.append(1)
        elif ("URU" in el):
            actual_vals.append(2)

    # predicted values
    predicted_vals = []
    for label in labels:
        if ("CAR" in label):
            predicted_vals.append(0)
        elif ("QUI" in label):
            predicted_vals.append(1)
        elif ("URU" in label):
            predicted_vals.append(2)

    # print(predicted_vals)
    # print(actual_vals)

    with open(output, "w+") as output_file:
        output_file.write("\n".join(labels))

    target_names = ["CAR", "QUI", "URU"]
    print(classification_report(actual_vals, predicted_vals))

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
        default="Data/Lab3/1-diff_features.txt",
        help="Output filename of labels when running predictions on a directory",
    )

    args = parser.parse_args()

    if args.sample:
        print(predict_shallow(args.sample))

    elif args.label_folder:
        predict_shallow_folder(args.label_folder, args.output)
