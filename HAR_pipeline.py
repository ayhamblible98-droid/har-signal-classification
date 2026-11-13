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

# ========================================================
# 5. MODEL COMPARISON (Logistic Regression vs RF vs GBM)
# ========================================================
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42
    ),
    'Gradient Boosting': HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, random_state=42
    ),
}

results = {}

print('\n' + '=' * 55)
print('         MODEL COMPARISON ON 40 DSP FEATURES         ')
print('=' * 55)

for name, model in models.items():
    print(f'Training {name}...')
    # Train model on scaled DSP features
    model.fit(X_train_scaled, y_train)

    # Evaluate on unseen test set
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

print('\n' + '-' * 45)
print(f"{'Classifier Model':<25} | {'Test Accuracy':<15}")
print('-' * 45)
for name, acc in results.items():
    print(f'{name:<25} | {acc * 100:.2f}%')
print('-' * 45)
