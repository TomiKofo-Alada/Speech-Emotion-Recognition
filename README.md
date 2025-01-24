# Speech Emotion Classification Using Advanced Machine Learning Models

## Introduction
This project explores the use of advanced machine learning models to classify emotions from speech signals, with a primary focus on a Convolutional Neural Network (CNN) architecture. The study evaluates the CNN model alongside four baseline models—CNN-LSTM, LSTM, Random Forest (RF), and Support Vector Machine (SVM)—on the RAVDESS dataset. Preprocessing techniques such as segmentation, silence removal, MFCC extraction, and normalization were employed to ensure high-quality input data.

---

## Features
- Preprocessing pipeline including:
  - Speech segmentation (270ms segments).
  - Silence removal to reduce redundant information.
  - Oversampling to balance class distribution.
  - MFCC feature extraction (40 coefficients).
  - Z-score normalization for standardized inputs.
- Primary model: Convolutional Neural Network (CNN) for spatial feature extraction.
- Baseline models for comparative analysis:
  - CNN-LSTM (hybrid spatial and temporal modeling).
  - LSTM (temporal modeling).
  - Random Forest (ensemble learning).
  - Support Vector Machine (margin-based classification).
- Evaluation metrics: Accuracy, F1-score, precision, recall, and confusion matrix.

---

## Technical Details
- **Frameworks and Libraries**:
  - TensorFlow (2.x) and Keras for deep learning models.
  - Scikit-learn for traditional machine learning models.
  - Librosa for audio feature extraction.
- **Hardware**: NVIDIA T4 GPU on Google Colab for training.
- **Hyperparameters**:
  - CNN: Adam optimizer, learning rate of 0.0001, batch size of 250, 1,000 epochs.
  - CNN-LSTM: RMSprop optimizer, learning rate of 0.0001, dropout (0.3), L2 regularization.
  - RF: 400 decision trees.
  - SVM: Radial basis function (RBF) kernel, cost parameter of 30.

---

## Dataset
- **Dataset Used**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).
  - Includes recordings from 24 actors (12 male, 12 female).
  - Eight emotional categories: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgusted, Surprised.
  - Audio-only subset used for classification.
- **Preprocessing Results**:
  - Original dataset: 1,440 speech signals.
  - After segmentation: 20,408 segments.
  - After silence removal: 9,580 segments.
  - After resampling: 12,056 balanced segments.

---

## Results
### Performance Comparison of Models
| Model       | Accuracy (%) | F1-Score |
|-------------|--------------|----------|
| CNN         | **80.0**     | 0.79     |
| CNN-LSTM    | 77.4         | 0.76     |
| LSTM        | 72.0         | 0.72     |
| Random Forest (RF) | 76.0         | 0.76     |
| Support Vector Machine (SVM) | 78.0         | 0.78     |

### Key Findings
- **CNN**: Achieved the highest accuracy (80.0%), excelling in classifying Neutral and Calm emotions but struggled with Disgust and Fearful.
- **Baseline Models**: Highlighted the strengths and limitations of hybrid and traditional approaches for spatial and temporal feature extraction.

### Comparative Analysis
- The CNN model achieved competitive results compared to models in related works:
  - CNN [32]: 82.0%
  - DCNN + CFS + SVM [6]: 81.3%
  - CNN [13]: 71.6%

---


## Getting Started

### Prerequisites
Ensure you have Python 3.8 or higher and the following libraries installed:
- TensorFlow
- Keras
- Scikit-learn
- Librosa
- Matplotlib


## Future Work
- Incorporate attention mechanisms or transformer-based architectures.
- Explore hybrid techniques like DCNN-CFS for improved feature selection.
- Optimize computational efficiency for real-time applications.

---

## Acknowledgments
- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Librosa Documentation](https://librosa.org/)

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
