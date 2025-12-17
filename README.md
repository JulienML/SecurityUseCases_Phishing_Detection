# Security Use Cases Final Project - Phishing Email Detection
A comparative study of classical Machine Learning vs Deep Learning approaches for phishing email detection.

## Overview

This project compares the effectiveness of traditional ML algorithms against modern DL architectures for identifying spam/phishing emails:
- **Machine Learning Models**: Logistic Regression, Random Forest, LinearSVC and XGBoost
- **Deep Learning Models**: RNN, LSTM and BERT

The goal is to evaluate which approach provides better performance for email security tasks.

## Dataset

The project uses the SpamAssassin public corpus dataset containing:
- **4,827 emails** (3,900 legitimate, 927 spam)
- Emails in raw text format with headers and body content
- Stored in `data/SpamAssasin.csv`

## Project Structure

```
├── data/
│   └── SpamAssasin.csv          # Email dataset
├── Machine Learning/
│   ├── ML Implementation.ipynb   # ML models implementation
│   └── preprocessing.py          # Data preprocessing functions
├── Deep Learning/
│   └── DL Implementation.ipynb   # Deep learning models
├── requirements.txt              # Python dependencies
└── README.md
```

## Installation

1. Clone the repository
    ```bash
    git clone https://github.com/JulienML/SecurityUseCases_Phishing_Detection.git
    cd SecurityUseCases_Phishing_Detection
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Machine Learning Approach
Open and run `Machine Learning/ML Implementation.ipynb` to:
- Preprocess the email data
- Train and evaluate classical ML models (Logistic Regression, Random Forest, LinearSVC, XGBoost)
- Compare performance across different ML algorithms

### Deep Learning Approach
Open and run `Deep Learning/DL Implementation.ipynb` to:
- Train and evaluate recurrent neural networks (RNN, LSTM, BERT)
- Compare performance across different DL architectures

### Comparing ML vs DL
Both notebooks provide detailed metrics to compare:
- Accuracy, Precision, Recall, F1-Score, AUC-ROC

## Results

The following table summarizes the best-performing Machine Learning models and each Deep Learning architecture, enabling a direct comparison across all evaluation metrics:

<table><thead>
  <tr>
    <th colspan="2" rowspan="2">Model</th>
    <th colspan="5">Metrics</th>
  </tr>
  <tr>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-Score</th>
    <th>ROC AUC</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="4">Machine Learning Models</td>
    <td>Logistic Regression</td>
    <td>0.9802</td>
    <td>0.9679</td>
    <td>0.9651</td>
    <td>0.9665</td>
    <td>0.9758</td>
  </tr>
  <tr>
    <td>Linear SVC</td>
    <td>0.9811</td>
    <td>0.9680</td>
    <td>0.9680</td>
    <td>0.9680</td>
    <td>0.9773</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.9759</td>
    <td>0.9702</td>
    <td>0.9477</td>
    <td>0.9588</td>
    <td>0.9677</td>
  </tr>
  <tr>
    <td>XGBoost Classifier</td>
    <td>0.9707</td>
    <td>0.9641</td>
    <td>0.9360</td>
    <td>0.9499</td>
    <td>0.9607</td>
  </tr>
  <tr>
    <td rowspan="3">Deep Learning Models</td>
    <td>RNN</td>
    <td>0.9707</td>
    <td>0.9613</td>
    <td>0.9390</td>
    <td>0.9500</td>
    <td>0.9615</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>0.9750</td>
    <td>0.9339</td>
    <td>0.9855</td>
    <td>0.9590</td>
    <td>0.9781</td>
  </tr>
  <tr>
    <td>BERT</td>
    <td>0.9802</td>
    <td>0.9573</td>
    <td>0.9767</td>
    <td>0.9669</td>
    <td>0.9792</td>
  </tr>
</tbody></table>

From a performance standpoint, both Machine Learning and Deep Learning models achieve high detection capabilities when applied to phishing detection. The results show that the performance gap between models is narrower than often assumed and depends primarily on the underlying representation rather than model complexity alone.

Machine Learning models achieve strong performance, particularly the Linear SVC with recall and ROC-AUC values comparable to those of Deep Learning models, demonstrating that welldesigned textual features can significantly enhance traditional classifiers.

Among Deep Learning approaches, sequence-based models exhibit heterogeneous behavior. While the LSTM model achieves the highest recall overall, transformer-based BERT delivers the most consistent performance across all metrics, including accuracy, F1-score, and ROC-AUC. This establishes BERT as the most reliable model in terms of global detection quality.