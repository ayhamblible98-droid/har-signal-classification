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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class_names = [
    'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
    'SITTING', 'STANDING', 'LAYING'
]

# Generate detailed metrics
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=class_names))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names, yticklabels=class_names
)
plt.title('Logistic Regression Confusion Matrix (18 Features)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# ==========================================
# 5. LIVE PREDICTION TEST (SINGLE INFERENCE)
# ==========================================
print("\n--- Live Prediction Test ---")

# Dictionary to map numerical IDs back to readable text
activity_map = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 
    4: 'SITTING', 5: 'STANDING', 6: 'LAYING'
}

# 1. Pick any random window from the test set (let's use index 42)
sample_index = 42

# 2. Isolate that single 128-timestep window. 
# We use slicing [42:43] instead of [42] so the array keeps its 3D shape (1, 128, 9)
# because our feature extractor expects a batch of signals.
single_raw_signal = X_test[sample_index:sample_index+1] 

# 3. Compress the 128 time-steps into our 18 baseline features
single_features = extract_baseline_features(single_raw_signal)

# 4. Ask the model to predict what activity this is
predicted_id = baseline_model.predict(single_features)[0]
actual_id = y_test[sample_index]

print(f"Analyzing sensor window #{sample_index}...")
print(f"-> Model Predicted: {activity_map[predicted_id]}")
print(f"-> Actual Truth:    {activity_map[actual_id]}")

# Try changing 'sample_index' to other numbers (like 0, 150, or 2000) 
# to test different activities!