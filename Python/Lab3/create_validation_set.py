"""
Write code that takes the labeled data and outputs a training dataset in `Data/Lab2/Train/`
and a validation dataset in `Data/Lab3/Validation`. These datasets should contain CSV files
that have their corresponding activity in their filename (similar to the the `Labeled/` folder).
No file should appear in both the training and validation sets.
Be sure to include documentation with clear instructions on how to run the script and expected outputs.
"""
import random, os

# create training directory
# current directory: /Users/carolyn_heinzer/Downloads/college things/2023-24 academics/spring/mobile computing/UChicago-Mobile-Computing-mc-labs-CHmakeupwork.git/Python
os.chdir('../../Data/Lab3')

# check if validation directory already exists; create if not
curr_path = os.getcwd()
dir_list = os.listdir(curr_path)

if 'Validation' not in dir_list:
    path = os.path.join(curr_path, 'Validation')
    os.mkdir(path)

if 'Train' not in dir_list:
    path = os.path.join(curr_path, 'Train')
    os.mkdir(path)

# choose unique random files from each group member
i = 0   # counter
used = []   # list to keep track of which files have been added to a dataset
for j in range(0, 2):
    while (0 <= i < 8):
        # actual random choice
        if (j == 0):
             curr_dir = "Carolyn_data"
        elif (j == 1):
             curr_dir = "Quinn_data"
        elif (j == 2):  
             curr_dir = "Urunna_data"
        # print(f)

        f = random.choice(os.listdir(curr_dir))

        if f not in used:

            # move the file to "Validation"; can just use the original name now
            src = os.getcwd() + "/" + curr_dir + "/" + f
            dst = os.getcwd() + "/Validation/" + curr_dir[0:3].upper() + f[3:]
            os.rename(src, dst)

            # add file name to used list (overall and this smaller one)
            used.append(f)
            i += 1  

# put the remaining files from "Labeled" into the Train set
for name in ("Carolyn_data", "Quinn_data", "Urunna_data"):
    for f in (os.listdir(name)):

            # move the file to "Train"; can just use the original name now
            src = os.getcwd() + "/" + name + "/" + f
            dst = os.getcwd() + "/Train/" + name[0:3].upper() + f[3:]
            os.rename(src, dst)

            # add file name to used list 
            used.append(f)

