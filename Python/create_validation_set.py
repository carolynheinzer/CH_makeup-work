"""
Write code that takes the labeled data and outputs a training dataset in `Data/Lab2/Train/`
and a validation dataset in `Data/Lab2/Validation`. These datasets should contain CSV files
that have their corresponding activity in their filename (similar to the the `Labeled/` folder).
No file should appear in both the training and validation sets.
Be sure to include documentation with clear instructions on how to run the script and expected outputs.
"""
import random, os

# create training directory
# current directory: /Users/carolyn_heinzer/Downloads/college things/2023-24 academics/spring/mobile computing/UChicago-Mobile-Computing-mc-labs-CHmakeupwork.git/Python
os.chdir('../Data/Lab2')

# check if validation directory already exists; create if not
curr_path = os.getcwd()
dir_list = os.listdir(curr_path)

if 'Validation' not in dir_list:
    path = os.path.join(curr_path, 'Validation')
    os.mkdir(path)

if 'Train' not in dir_list:
    path = os.path.join(curr_path, 'Train')
    os.mkdir(path)

# choose unique random files from 'Labeled'
i = 0   # counter
used = []   # list to keep track of which files have been added to a dataset
while (0 <= i < 24):
    # actual random choice
    f = random.choice(os.listdir("Labeled"))
    # print(f)

    if f not in used:
        """# copy contents of the random file 
        os.chdir("Labeled")
        f_opened = open(f, "r")
        f_contents = f_opened.read()
        f_opened.close()
        os.chdir("..")
        f_lines = f_contents.split("\n")

        # create new file with temp different name to distinguish it from the version of it in "Labeled"
        new_name = f[:-4] + "_val.csv"
        new_f = open(new_name, "w")
        new_f.writelines(f_lines)
        new_f.close()"""

        # move the file to "Validation"; can just use the original name now
        src = os.getcwd() + "/Labeled/" + f
        dst = os.getcwd() + "/Validation/" + f
        os.rename(src, dst)

        # add file name to used list 
        used.append(f)
        i += 1  

# put the remaining files from "Labeled" into the Train set
for f in (os.listdir("Labeled")):
    if f not in used:
        """# open f and copy its contents
        os.chdir("Labeled")
        f_opened = open(f, "r")
        f_contents = f_opened.read()
        f_opened.close()
        os.chdir("..")
        f_lines = f_contents.split("\n")

        # create new file with temp different name to distinguish it from the version of it in "Labeled"
        new_name = f[:-4] + "_train.csv"
        new_f = open(new_name, "w")
        new_f.writelines(f_lines)
        new_f.close()"""

        # move the file to "Train"; can just use the original name now
        src = os.getcwd() + "/Labeled/" + f
        dst = os.getcwd() + "/Train/" + f
        os.rename(src, dst)

        # add file name to used list 
        used.append(f)

