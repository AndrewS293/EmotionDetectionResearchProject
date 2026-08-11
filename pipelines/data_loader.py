import os
import pickle
import numpy as np
from scipy.signal import resample

VALID_LABELS = [0,1,2,3]


#Loads all the data from the file

def load_subject(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist = data["signal"]["wrist"]

    eda = wrist["EDA"].flatten()
    bvp = wrist["BVP"].flatten()
    temp = wrist["TEMP"].flatten()
    acc = wrist["ACC"]

    target_len = len(eda)

    bvp = resample(bvp, target_len)

    acc_x = resample(acc[:,0], target_len)
    acc_y = resample(acc[:,1], target_len)
    acc_z = resample(acc[:,2], target_len)

    labels = data["label"]
    labels = resample(labels, target_len)
    labels = np.round(labels).astype(int)

    mask = np.isin(labels, VALID_LABELS)

    labels = labels[mask]

    combined = np.column_stack([
        eda[mask],
        bvp[mask],
        temp[mask],
        acc_x[mask],
        acc_y[mask],
        acc_z[mask]
    ])

    return combined, labels


def load_all_subjects(data_dir):
    all_X = []
    all_y = []

    for file in os.listdir(data_dir):
        if file.endswith('.pkl'):
            path = os.path.join(data_dir, file)
            print(path)

            X, y = load_subject(path)

            all_X.append(X)
            all_y.append(y)

    return all_X, all_y

#These are the variables you'll want to use. They are the raw signals, just flattened and resampled. 
#I believe they are dataframes so they should work, you can double chack though. Let me know if you need help or you need them modified in some way
#all_X contains all the physiological signals, and all_y contains the labels for each of those signals. They are in the same order