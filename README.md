# Bank Marketing Prediction Model

A from-scratch logistic regression model to predict whether bank customers will subscribe to a term deposit using customer demographics and campaign data.

## Project Overview
This project implements logistic regression from the ground up using only NumPy and Pandas to predict bank marketing success. The model predicts customer subscription likelihood based on demographics, financial status, and previous marketing interactions.

## Dataset
- **Source**: Bank Marketing Dataset (UCI ML Repository)
- **Original size**: 45,211 customers
- **Features**: 17 original features
- **Target**: Binary classification (will customer subscribe?)
- **Class Distribution**: 
  - No subscription: 39,922 (88.3%)
  - Subscription: 5,289 (11.7%)

## Features Used
### Original Features
1. **Demographics**: Age, job, marital status, education
2. **Financial**: Account balance, housing loan, personal loan, default history
3. **Campaign Data**: Contact method, call duration, campaign frequency
4. **Previous Campaigns**: Days since last contact, previous outcome, number of contacts

### Engineered Features (~35 total)
1. **Improved Ordinal Encoding**: Education levels, contact methods, previous outcomes
2. **Smart Binning**: Age groups, balance quartiles, call duration categories
3. **Interaction Features**: Age×Balance, Duration×Campaign, Previous×Pdays
4. **Business Features**: Financial stability, debt burden, contact history
5. **Individual Job Types**: 12 separate dummy variables for each profession

## Data Preprocessing
- Converted categorical variables to numerical encodings
- Created ordinal rankings for education (primary < secondary < tertiary)
- Generated interaction features for complex relationships
- Scaled numerical features to 0-1 range for consistent learning
- Preserved class imbalance for realistic evaluation
- Split data 80/20 for training and testing

## Model Implementation
Built entirely from scratch without scikit-learn:

### Logistic Regression Equation
```
probability = sigmoid(bias + w1×feature1 + w2×feature2 + ... + wn×featuren)
prediction = 1 if probability >= threshold else 0
```

### Key Functions
- **Sigmoid**: Overflow-protected logistic function
- **Cost Function**: Cross-entropy loss with epsilon clipping
- **Gradients**: Partial derivatives for weight updates
- **Gradient Descent**: Iterative parameter optimization

### Training Parameters
- **Learning Rate**: 0.02
- **Iterations**: 1,000
- **Optimization**: Gradient descent with cross-entropy loss
- **Threshold**: 0.15 (optimized for F1 score)

## Results

### Model Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 83.81% |
| **Precision** | 37.48% |
| **Recall** | 56.44% |
| **F1 Score** | 45.05% |

### Confusion Matrix
```
                 Predicted
              No    Yes
Actual  No   6979  1001
        Yes   463   600
```

### Training Progress
- **Initial Cost**: 0.693147
- **Final Cost**: 0.347245
- **Convergence**: Smooth cost reduction over 1,000 iterations

## Key Findings
1. **Housing loans strongly negative**: Largest negative weight (-0.479)
2. **Contact method matters**: Cellular contact increases subscription probability
3. **Job type significance**: Management and student roles show different patterns
4. **Duration importance**: Longer calls correlate with better outcomes
5. **Feature engineering impact**: 60% improvement in F1 score from baseline

## Technical Skills Demonstrated
- **Advanced Feature Engineering**: Creating meaningful interactions and derived features
- **Mathematical Implementation**: Sigmoid function, gradient computation, cost optimization
- **Class Imbalance Handling**: Threshold optimization for skewed datasets
- **Data Preprocessing**: Ordinal encoding, scaling, categorical variable handling
- **Model Evaluation**: Comprehensive metrics analysis for binary classification

## Files
- `bank_logistic.ipynb`: Complete analysis and model implementation
- `bank-full.csv`: Original bank marketing dataset
- Data preprocessing, feature engineering, model training, and evaluation

## Business Impact
- **Marketing Efficiency**: 37% success rate vs 12% random targeting
- **Campaign Optimization**: Target 1,601 customers to get ~600 subscriptions
- **Cost Effectiveness**: Reduces wasted marketing spend by 3x
- **Customer Insights**: Identifies key factors driving subscription behavior

## Future Improvements
- Implement SMOTE or class weighting for better imbalance handling
- Add L1/L2 regularization to prevent overfitting
- Cross-validation for more robust performance estimates
- Feature selection using correlation analysis
- Ensemble methods (Random Forest, XGBoost) for comparison

## Dependencies
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Usage
1. Load and explore the bank marketing dataset
2. Engineer features with ordinal encoding and interactions
3. Split data and scale features appropriately
4. Train logistic regression model with gradient descent
5. Optimize threshold for best F1 score
6. Evaluate performance and analyze feature importance

## Conclusion
Successfully built a working customer subscription prediction system achieving 45% F1 score on imbalanced banking data. The from-scratch implementation demonstrates solid understanding of logistic regression fundamentals, advanced feature engineering techniques, and practical machine learning skills for real-world business applications.
