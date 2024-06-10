"""
Write code that takes the labeled data and outputs a training dataset in `Data/Lab2/Train/`
and a validation dataset in `Data/Lab3/Validation`. These datasets should contain CSV files
that have their corresponding user in their filename (similar to the the `Labeled/` folder).
No file should appear in both the training and validation sets.
Be sure to include documentation with clear instructions on how to run the script and expected outputs.
"""
import random, os

# create training directory
# current directory: /Users/carolyn_heinzer/Downloads/college things/2023-24 academics/spring/mobile computing/UChicago-Mobile-Computing-mc-labs-CHmakeupwork.git/Python
os.chdir('Data/Lab3')

# check if validation directory already exists; create if not
curr_path = os.getcwd()
dir_list = os.listdir(curr_path)

if 'AltTest' not in dir_list:
    path = os.path.join(curr_path, 'AltTest')
    os.mkdir(path)

if 'AltTrain' not in dir_list:
    path = os.path.join(curr_path, 'AltTrain')
    os.mkdir(path)

# choose unique random files from each group member
i = 0   # counter
used = []   # list to keep track of which files have been added to a dataset
directories = ["Carolyn_data", "Quinn_data", "Urunna_data"]

"""
while (0 <= i < 18):
    if (0 <= i < 6):
        curr_dir = "Carolyn_data"
    elif (6 <= i < 12):
        curr_dir = "Quinn_data"
    elif (12 <= i < 18):
        curr_dir = "Urunna_data" """

while (0 <= i < 15):
    if (0 <= i < 4):
        curr_dir = "Carolyn_data"
    elif (5 <= i < 9):
        curr_dir = "Quinn_data"
    elif (10 <= i < 14):
        curr_dir = "Urunna_data"

    f = random.choice(os.listdir(curr_dir))

    if f not in used:
        src = os.getcwd() + "/" + curr_dir + "/" + f
        dst = os.getcwd() + "/AltTrain/" + f
        os.rename(src, dst)

        # add file name to used list (overall and this smaller one)
        used.append(f)
        i += 1  

# put the remaining files from "Labeled" into the Train set
for directory in directories:
    for f in (os.listdir(directory)):

            # move the file to "Train"
            src = os.getcwd() + "/" + directory + "/" + f
            dst = os.getcwd() + "/AltTest/" + f
            os.rename(src, dst)

            # add file name to used list 
            used.append(f)

