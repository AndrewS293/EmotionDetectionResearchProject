import os
import numpy as np
import pandas as pd
from scipy.signal import resample



EEG_COLUMNS = [
    "AF3", "F7", "F3", "FC5",
    "T7", "P7", "O1", "O2",
    "P8", "T8", "FC6", "F4",
    "F8", "AF4"
]




def tag_to_class(valence, arousal):

    if valence >= 5 and arousal >= 5:
        return 0 #Happy

    elif valence < 5 and arousal < 5:
        return 1 #Sad

    elif valence >= 5 and arousal < 5:
        return 2 #Calm

    elif valence < 5 and arousal >= 5:
        return 3 #Angry




def load_subject(signal_path, label_path):

    signal_df = pd.read_csv(signal_path)

    label_df = pd.read_csv(label_path)


    eda = signal_df["GSR"].to_numpy() #gsr

    bvp = (
        signal_df["ECG_Right"].to_numpy()
        - signal_df["ECG_Left"].to_numpy()
    )  #ecg


    n = len(eda) // 8

    eda = resample(eda, n)
    bvp = resample(bvp, n)

    combined = np.column_stack([

        eda,        # EDA equivalent

        bvp

    ])


    valence = float(label_df.loc[0, "Valence"])
    arousal = float(label_df.loc[0, "Arousal"])


    emotion = tag_to_class(valence, arousal)

    labels = np.full(n, emotion)

    return combined, labels



def load_all_subjects(data_dir):

    all_X = []
    all_y = []

    print("Dataset directory:", data_dir)

    for user in sorted(os.listdir(data_dir)):

        #print("\nUser folder:", user)

        user_folder = os.path.join(data_dir, user)

        if not os.path.isdir(user_folder):
            #print("Not a directory")
            continue

        label_folder = os.path.join(user_folder, "Label")

        #print("Label folder exists:", os.path.exists(label_folder))

        for file in sorted(os.listdir(user_folder)):

            #print("Found:", file)

            if not file.endswith(".csv"):
                continue

            signal_path = os.path.join(user_folder, file)

            base = file.replace(".csv", "")

            user, video = base.split("_")
            
            label_name = f"{user}_selfAss_{video}.csv"

            label_path = os.path.join(label_folder, label_name)

            #print("Looking for:", label_name)

            if not os.path.exists(label_path):
                #print("Missing label!")
                continue

            X, y = load_subject(signal_path, label_path)

            #print("Loaded:", X.shape, y.shape)

            all_X.append(X)
            all_y.append(y)

    #
    # print("\nSubjects loaded:", len(all_X))

    return all_X, all_y