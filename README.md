# 🧠 Mental Health Prediction using Neural Networks

This project focuses on predicting mental health conditions using a **simple feed-forward neural network** built with TensorFlow and Keras. By analyzing structured survey data, the model helps classify individuals who might be at risk for mental health issues based on various personal and workplace factors.


## 📌 Project Objective

- **Build and train** a feed-forward neural network.
- **Classify structured data** related to mental health.
- Use **real-world survey data** to predict if someone is likely to have a mental health condition.
- Perform **hyperparameter tuning** to improve model performance.
- Visualize results using **accuracy and loss curves**.

## 📂 Dataset

- **Source**: [Kaggle - Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- Contains information like:
  - Age, gender, work environment, family history, and more.
  - Target: Whether the individual has a mental health condition.


## 🛠 Tools & Libraries Used

- Python
- TensorFlow / Keras
- pandas
- scikit-learn
- matplotlib
- seaborn

## 🔍 Project Workflow

### 1. **Data Loading & Preprocessing**
- Loaded structured survey data.
- Handled missing values, dropped irrelevant columns.
- Converted categorical variables using label encoding/one-hot encoding.
- Scaled numeric features for better model performance.

### 2. **Neural Network Design**
- Simple feed-forward neural network (Sequential Model)
- Architecture:
  - Input Layer → Dense (ReLU) → Hidden Layer → Output Layer (Sigmoid)

### 3. **Model Training & Evaluation**
- Used **binary cross-entropy** as the loss function.
- Evaluated using **accuracy and loss curves** on training and test data.
- Final accuracy: ~70%

### 4. **Hyperparameter Tuning**
- Experimented with:
  - Learning rate
  - Batch size
  - Number of epochs
  - Dropout rate
- Best results achieved with:
  - 32 neurons in hidden layer
  - Dropout: 0.3
  - Batch size: 32
  - Epochs: 50

## 📈 Results

- **Final Test Accuracy**: ~70.24%
- **Final Test Loss**: ~0.60
- Plotted training vs. validation accuracy and loss
- Model performs decently for a basic feed-forward neural net

## ✅ Advantages & Importance

- **Early detection** of mental health issues through data
- Can assist HR departments or wellness teams in identifying patterns
- Encourages the use of AI in promoting mental health awareness
- Real-world application of ML in healthcare and workplace settings

## 🚀 Future Improvements

- Use more complex models like RNNs or transformers with NLP text inputs.
- Integrate with mobile/web apps for real-time screening.
- Add more robust features like sentiment analysis from user feedback.
- Train with larger, more diverse datasets to generalize better.

## 🙌 Acknowledgements

 [Kaggle Dataset](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

> “Tech for good – Let’s make mental health care accessible with the power of AI.” 💙
