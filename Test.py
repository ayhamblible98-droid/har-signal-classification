import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Label Mapping
ACTIVITY_NAMES = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING', 5: 'STANDING', 6: 'LAYING'
}

# 2. Load 3-Axis Signals & Target Labels
base_path = 'UCI HAR Dataset/UCI HAR Dataset/train/'
acc_x = pd.read_csv(
    base_path + 'Inertial Signals/body_acc_x_train.txt',
    delim_whitespace=True,
    header=None,
).values
acc_y = pd.read_csv(
    base_path + 'Inertial Signals/body_acc_y_train.txt',
    delim_whitespace=True,
    header=None,
).values
acc_z = pd.read_csv(
    base_path + 'Inertial Signals/body_acc_z_train.txt',
    delim_whitespace=True,
    header=None,
).values
labels = pd.read_csv(
    base_path + 'y_train.txt', delim_whitespace=True, header=None
).values.ravel()

# 3. Create Stacked Subplots (6 Activities x 3 Axes)
fig, axes = plt.subplots(6, 3, figsize=(15, 12), sharey=True)

for act_id in range(1, 7):
    # Find the index of the first window matching this activity
    sample_idx = np.where(labels == act_id)[0][0]
    row = act_id - 1

    axes[row, 0].plot(acc_x[sample_idx], color='r')
    axes[row, 0].set_ylabel(ACTIVITY_NAMES[act_id], fontsize=9)
    axes[row, 1].plot(acc_y[sample_idx], color='g')
    axes[row, 2].plot(acc_z[sample_idx], color='b')

    if row == 0:
        axes[row, 0].set_title('X-Axis Acceleration')
        axes[row, 1].set_title('Y-Axis Acceleration')
        axes[row, 2].set_title('Z-Axis Acceleration')

plt.xlabel('Time Step (0 to 127 @ 50 Hz)')
plt.tight_layout()
plt.show()