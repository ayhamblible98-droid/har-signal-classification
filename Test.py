import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

base_dir = 'UCI HAR Dataset/UCI HAR Dataset/'

# 1. Load all 9 inertial signal channels into shape (N, 128, 9)
def load_split_3d(dataset_path, split):
    signals = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    channels = []
    for sig in signals:
        file_path = f"{dataset_path}{split}/Inertial Signals/{sig}_{split}.txt"
        # Using sep=r'\s+' handles variable-length whitespace delimiters
        data = pd.read_csv(file_path, sep=r'\s+', header=None).values
        channels.append(data)
    
    return np.stack(channels, axis=-1)

# 2. Load Train/Test sets & Targets
print("Loading signal data...")
X_train = load_split_3d(base_dir, 'train')  # (7352, 128, 9)
X_test  = load_split_3d(base_dir, 'test')   # (2947, 128, 9)

y_train = pd.read_csv(base_dir + 'train/y_train.txt', sep=r'\s+', header=None).values.ravel()
y_test  = pd.read_csv(base_dir + 'test/y_test.txt', sep=r'\s+', header=None).values.ravel()

# 3. Step 3: Compute basic statistics (Mean & Std across axis=1)
def extract_baseline_features(X_3d):
    means = np.mean(X_3d, axis=1)  # (N, 9)
    stds  = np.std(X_3d, axis=1)   # (N, 9)
    return np.hstack([means, stds]) # (N, 18)

X_train_baseline = extract_baseline_features(X_train)
X_test_baseline  = extract_baseline_features(X_test)

print(f"X_train_baseline shape: {X_train_baseline.shape}")
print(f"X_test_baseline shape:  {X_test_baseline.shape}")

# 4. Step 4: Train Baseline Logistic Regression
print("Training Logistic Regression...")
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train_baseline, y_train)

# Evaluate accuracy
y_pred = baseline_model.predict(X_test_baseline)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBaseline Test Accuracy: {accuracy * 100:.2f}%")