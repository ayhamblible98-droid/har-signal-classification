# HAR Project Experimental Results

## Feature Ablation Study (Logistic Regression)

| Feature Subset | Test Accuracy |
|---|---|
| mean + std (18) | 78.52% |
| + magnitude (22) | 83.68% |
| + dominant freq (31) | 87.31% |
| + spectral energy (40) | 87.61% |

## Model Comparison (40 DSP Features)

| Classifier Model | Test Accuracy |
|---|---|
| Logistic Regression | 87.61% |
| Random Forest | 88.33% |
| Gradient Boosting | 89.62% |

Confusion Matrix: 
<img width="1200" height="900" alt="image" src="https://github.com/user-attachments/assets/58069072-8868-4a6f-ad42-e2db01c86644" />
