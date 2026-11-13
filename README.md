# Human Activity Recognition from Smartphone Inertial Signals

Classifying six everyday activities from accelerometer and gyroscope signals, using signal-processing features (time domain, acceleration magnitude, FFT) and classical machine learning.

> Built as an electrical-engineering-focused project: the goal is to show how physically motivated features (gravity orientation, oscillation strength, step frequency) perform compared with basic statistics alone.

---

## 1. Problem

Given a 2.56-second window of smartphone sensor data, predict which activity the person is doing:

`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`

This is a supervised classification problem on multichannel time-series data.

## 2. Dataset

**UCI Human Activity Recognition Using Smartphones** (Anguita et al., 2013).

- 30 volunteers wearing a smartphone on the waist
- 9 inertial channels sampled at 50 Hz: body acceleration (x, y, z), gyroscope (x, y, z), total acceleration (x, y, z)
- Windows of 128 readings (2.56 s) with 50% overlap
- Official split by subject: 21 people for training (7352 windows), 9 for testing (2947 windows)

Download: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The dataset is **not** included in this repository. Place it so the folder layout is:

```
UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/...
UCI HAR Dataset/UCI HAR Dataset/test/Inertial Signals/...
```

Note: the dataset authors already applied noise filtering and separated body motion from gravity before release.

## 3. Method

### Pipeline

```
raw windows (N, 128, 9)  ->  40 engineered features  ->  StandardScaler  ->  classifier  ->  activity
```

### Features (40 per window)

| Group | Count | Physical meaning |
|---|---|---|
| Mean per channel | 9 | Average level; for total acceleration it reveals gravity direction (device orientation) |
| Standard deviation per channel | 9 | Amount of movement (flat for sitting, large for walking) |
| Magnitude mean (body and total acceleration) | 2 | Orientation-independent signal level |
| Magnitude std (body and total acceleration) | 2 | Orientation-independent movement strength |
| Dominant frequency per channel (FFT, DC removed) | 9 | Step rhythm / cadence |
| Spectral energy per channel (FFT) | 9 | Overall signal power at all frequencies |

Features are standardized with `StandardScaler`, fitted on the training set only to avoid information leakage into the test set.

### Models compared

- Logistic Regression
- Random Forest (100 trees, max depth 15)
- Histogram Gradient Boosting (100 iterations)

All use `random_state=42`.

## 4. Results

### 4.1 Model comparison (40 features, official subject-wise test set)

| Model | Test accuracy |
|---|---|
| Logistic Regression | 87.61% |
| Random Forest | 88.33% |
| Gradient Boosting | 89.62% |

The three models are within about two percentage points of each other. With ~2,950 test windows and a single run per model, small differences should be read with caution.

### 4.2 Feature ablation (Logistic Regression, scaled)

Features were added group by group to measure what each contributes.

<!-- TODO: replace the "?" values with the numbers printed by section 7 of the script -->

| Feature set | Test accuracy |
|---|---|
| Mean + std (18 features) | 78.52% |
| + Magnitude (22 features) | 83.68% |
| + Dominant frequency (31 features) | 87.31% |
| + Spectral energy (40 features) | 87.61% |

> An early unscaled baseline (mean + std, 18 features) reached 76.52%. It is not directly comparable to the table above because it used no feature scaling. The ablation table is the fair comparison.

### 4.3 Per-class performance (Gradient Boosting)

<!-- TODO: paste the classification report printed by section 8 of the script -->

```
(paste classification report here)
--- Classification Report (Gradient Boosting) ---
              precision    recall  f1-score   support

     WALKING       0.91      0.90      0.91       496
    UPSTAIRS       0.87      0.89      0.88       471
  DOWNSTAIRS       0.88      0.87      0.87       420
     SITTING       0.88      0.80      0.84       491
    STANDING       0.83      0.90      0.86       532
      LAYING       1.00      1.00      1.00       537

    accuracy                           0.90      2947
   macro avg       0.90      0.89      0.89      2947
weighted avg       0.90      0.90      0.90      2947

```
### 4.4 Confusion matrix

![Normalized confusion matrix](results.md)

## 5. Analysis

<!-- TODO: write 4-6 sentences IN YOUR OWN WORDS, based on the tables above. Suggested questions:
     - Which feature group gave the biggest improvement in the ablation, and did any group not help?
     - Which activities are most often confused in the confusion matrix?
     - Why might those pairs be hard (think: phone on the waist, similar orientation, similar rhythm)?
     - Why does a simple linear model come close to the stronger models?
-->

(Write your interpretation here.)

## 6. Limitations

- **Single train/test split and one run per model.** No error bars; the ranking of models is not statistically established.
- **Pre-filtered data.** The authors already filtered the signals, so low-pass filtering adds little at 50 Hz sampling (Nyquist frequency is 25 Hz).
- **Fixed sensor placement.** All data comes from a waist-mounted phone; results will not transfer to other positions or devices without retraining.
- **Small population.** 30 volunteers, mostly young adults performing scripted activities in a lab-like setting.
- **No real-time evaluation.** Windows are evaluated independently, without temporal smoothing across consecutive windows.

## 7. How to run

```bash
# 1. Clone and enter the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset (see section 2) and place it in the project folder

# 4. Run
python har_pipeline.py
```

The script prints the model comparison, the ablation table and the classification report, and saves `confusion_matrix.png`.

### Predicting on your own window

A window must have shape `(128, 9)` with channels in this order, at 50 Hz and in the dataset's units (accelerations in g, gyroscope in rad/s):

`body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z`

## 8. Project structure

```
.
├── har_pipeline.py
├── requirements.txt
├── README.md
└── results/
    └── confusion_matrix.png
```

## 9. Future work

- Subject-wise cross-validation (`GroupKFold`) for more reliable estimates
- Multiple random seeds with mean and standard deviation
- A small 1D CNN on the raw windows for comparison with the feature-based models
- Testing on data recorded with a different phone to measure domain shift

## 10. Development note

The code was written with AI assistance (Gemini). I designed the experiments, reviewed and ran the code, and analyzed the results.

## References

- Anguita, D., Ghio, A., Oneto, L., Parra, X., Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013.

## License

MIT
