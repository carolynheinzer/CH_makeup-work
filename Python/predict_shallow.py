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

"""
Create a non-deep learning classifier (e.g. multiclass SVM, decision tree, random forest)
to perform activity detection that improves upon your prior algorithm.

Usage:
    
    python3 Python/predict_shallow.py <sensor .csv sample>

    python3 Python/predict_shallow.py --label_folder <folder with sensor .csv samples>
"""

def extract(file_name: str, test_train: int):
    # a = (v_f - v_i)/t
    # returns dict of x, y, z velocities and time
    if test_train == 0:
        name = "Data/Lab2/Train/" + file_name
    elif test_train == 1:
        name = file_name
    elif test_train == 2:
        name = "Data/Lab2/Validation/" + file_name
    f = open(name, "r")

    # skip first line
    line1 = f.readline()

    # initialize dict of x, y, and z eadset_angularVel.z","headset_pos.x","headset_pos.y","headset_pos.z","headset_rot.x","headset_rot.y","headset_rot.z","controller_left_vel.x","controller_left_vel.y","controller_left_vel.z","controller_left_angularVel.x","controller_left_angularVel.y","controller_left_angularVel.z","controller_left_pos.x","controller_left_pos.y","controller_left_pos.z","controller_left_rot.x","controller_left_rot.y","controller_left_rot.z","controller_right_vel.x","controller_right_vel.y","controller_right_vel.z","controller_right_angularVel.x","controller_right_angularVel.y",velocities + time
    info = {}
    info['time'] = []
    info['x'] = []
    info['y'] = []
    info['z'] = []

    # data: time,headset_vel.x,headset_vel.y,headset_vel.z,headset_angularVel.x,headset_angularVel.y,headset_angularVel.z,headset_pos.x,headset_pos.y,headset_pos.z,headset_rot.x,headset_rot.y,headset_rot.z,controller_left_vel.x,controller_left_vel.y,controller_left_vel.z,controller_left_angularVel.x,controller_left_angularVel.y,controller_left_angularVel.z,controller_left_pos.x,controller_left_pos.y,controller_left_pos.z,controller_left_rot.x,controller_left_rot.y,controller_left_rot.z,controller_right_vel.x,controller_right_vel.y,controller_right_vel.z,controller_right_angularVel.x,controller_right_angularVel.y,controller_right_angularVel.z,controller_right_pos.x,controller_right_pos.y,controller_right_pos.z,controller_right_rot.x,controller_right_rot.y,controller_right_rot.z
    # want time, controller_left_vel.x, controller_left_vel.y, controller_left_vel.z
    # indices [0, 13, 14, 15]

    for line in f.readlines():
        lst = line.split(",")
        info['time'].append(lst[0])
        info['x'].append(lst[13])
        info['y'].append(lst[14])
        info['z'].append(lst[15])

    f.close()
    return info

def preprocess(file_name: str, test_train: int):
    # finds the mean, standard deviation, and number of peaks
    # returns dict containing all relevant column values for dataframe
    info = extract(file_name, test_train)

    # initialize results variable
    res = {}

    # x values - time
    times = list(map(float, info['time']))

    # y values - 3 sets (x, y, and z components of velocity)
    v_x = list(map(float, info['x']))
    v_y = list(map(float, info['y']))
    v_z = list(map(float, info['z']))

    # find the numbers of peaks in these sets of y values
    peaks_x, _ = find_peaks(v_x)
    peaks_y, _ = find_peaks(v_y)
    peaks_z, _ = find_peaks(v_z)

    res['x_peaks'] = len(peaks_x)
    res['y_peaks'] = len(peaks_y)
    res['z_peaks'] = len(peaks_z)

    # find mean of x, y, and z components of velocity
    # create the series
    s_x = pandas.Series(data = v_x)
    s_y = pandas.Series(data = v_y)
    s_z = pandas.Series(data = v_z)

    res['x_mean'] = s_x.mean()
    res['y_mean'] = s_y.mean()
    res['z_mean'] = s_z.mean()

    # find standard deviation for x, y, and z components of velocity
    res['x_std'] = s_x.std()
    res['y_std'] = s_y.std()
    res['z_std'] = s_z.std()

    # activity type
    res['code'] = file_name[0:3]

    return res

def combine_files(dir: str, file_out: str, code: int):
    # create a dataframe from all files in Train
    # print(os.getcwd())
    # files = os.listdir("Data/Lab2/Train")
    data_dicts = []

    for file in dir:
        data_dict = preprocess(file, code)
        data_dicts.append(data_dict)

    # field names
    fields = ['x_peaks', 'y_peaks', 'z_peaks', 'x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'code']

    # name of csv
    filename = file_out

    # write to csv
    with open(filename, 'w') as csvfile:
        # create a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # write header and data rows
        writer.writeheader()
        writer.writerows(data_dicts)

def create_dtree(file_name: str):
    # create dataframe and ensure all data is numbers
    d = {'STD': 0, 'SIT': 1, 'JOG': 2, 'ARC': 3, 'STR': 4, 'DRI':5, 'TWS':6}
    df = pandas.read_csv(file_name)
    
    df['code'] = df['code'].map(d)
    # print(df)

    # separate feature columns from target column
    features = ['x_peaks', 'y_peaks', 'z_peaks', 'x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std']

    X = df[features].values
    y = df['code']

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)
    return dtree

def predict_shallow(sensor_data: str) -> str:
    # Run prediction on an sensor data sample.

    # Replace the return value of this function with the output activity label
    # of your shallow classifier for the given sample. Feel free to load any files and write
    # helper functions as needed.

    dtree = create_dtree("data_summary.csv")

    # extract the input for the prediction
    ipt = preprocess(sensor_data, 1)
    lst_ipt = []

    for key, val in ipt.items():
        if (key != 'code'):
            lst_ipt.append(val)

    code = dtree.predict([lst_ipt])     

    result = 100    # garbage value

    if (code == 0):
        result = 'STD'
    elif (code == 1):
        result = 'SIT'
    elif (code == 2):
        result = 'JOG'
    elif (code == 3):
        result = 'ARC'
    elif (code == 4):
        result = 'STR'
    elif (code == 5):
        result = 'DRI'
    elif (code == 6):
        result = 'TWS'

    return result

def predict_shallow_folder(data_folder: str, output: str):
    # Run the model's prediction on all the sensor data in data_folder, writing labels
    # in sequence to an output text file.

    # create data summary
    files = os.listdir("Data/Lab2/Train")
    out_file = "data_summary.csv"
    combine_files(files, out_file, 0)

    data_files = sorted(glob(f"{data_folder}/*.csv"))
    data_files = [f for f in data_files if 'data_summary' not in f]
    # print(data_files)
    labels = map(predict_shallow, data_files)

    with open(output, "w+") as output_file:
        output_file.write("\n".join(labels))

def calc_accuracy():
    # accuracy and precision reports
    val_files = os.listdir("Data/Lab2/Validation")
    combine_files(val_files, "val_data_summary.csv", 2)

    test = []
    pred = []

    for file in val_files:
        test.append(predict_shallow("Data/Lab2/Validation/" + file))
        pred.append(file[0:3])

    # convert to ints
    test_vals = []
    pred_vals = []

    for arr in test, pred:
        for el in arr:
            if ('STD' in el):
                test_vals.append(0)
                pred_vals.append(0)
            elif ('SIT' in el):
                test_vals.append(1)
                pred_vals.append(1)
            elif ('JOG' in el):
                test_vals.append(2)
                pred_vals.append(2)
            elif ('ARC' in el):
                test_vals.append(3)
                pred_vals.append(3)
            elif ('STR' in el):
                test_vals.append(4)
                pred_vals.append(4)
            elif ('DRI' in el):
                test_vals.append(5)
                pred_vals.append(5)
            elif ('TWS' in el):
                test_vals.append(6)
                pred_vals.append(6)

    target_names = ['STD', 'SIT', 'JOG', 'ARC', 'STR', 'DRI', 'TWS']
    print(classification_report(pred_vals, test_vals, target_names=target_names))

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
        default="Data/Lab2/Labels/shallow.txt",
        help="Output filename of labels when running predictions on a directory",
    )

    args = parser.parse_args()

    if args.sample:
        print(predict_shallow(args.sample))

    elif args.label_folder:
        predict_shallow_folder(args.label_folder, args.output)

    calc_accuracy()