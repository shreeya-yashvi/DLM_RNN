# 🎥 Reviews Sentiment Analysis using Recurrent Neural Network (RNN)

### Contributors:
- **Shreeya Yashvi (055045)**  
- **Kashish Srivastava (055046)**  

---

## 🎯 Objective
The objective is to design, implement, and evaluate a deep learning sentiment analysis model using an RNN architecture. This model classifies movie reviews by capturing sequential patterns in text data and determines whether a review is **positive** or **negative**.

---

## 🔥 Problem Statement
- Online movie reviews play a critical role in shaping public perception.
- Sentiment classification is inherently complex due to nuances in language.
- An RNN-based approach is used to capture contextual information and classify reviews effectively.

---

## 🌟 Key Features
### 1. Data Preprocessing
- **Sentiment Encoding:**  
    - Positive Sentiment → **1**  
    - Negative Sentiment → **0**  
- **Text Normalization:**  
    - Removing special characters, lowercasing, and cleaning text.
- **Tokenization:**  
    - Splitting text into tokens using a vocabulary of the top **20,000** most frequent words.
- **Sequence Padding:**  
    - Padding/truncating sequences to a fixed length of **400** tokens for consistency.

### 2. Model Development
- **Embedding Layer:**  
    - Input dimension: 20,000  
    - Output dimension: 128  
    - Input length: 400  

- **Recurrent Layer:**  
    - SimpleRNN with 64 units  
    - Tanh activation with dropout of 0.2  

- **Fully Connected Layer:**  
    - Dense layer with 1 neuron  
    - Sigmoid activation for binary classification  

### 3. Training & Evaluation
- **Training Dataset:**  
    - IMDB dataset with **50,000 reviews** (Sampled 40,000 reviews with a random state of 4546).
    - Split into **80% training** and **20% testing**.
- **Testing Dataset:**  
    - 151 reviews manually scraped from Metacritic for cross-validation.
- **Model Compilation:**  
    - Loss: Binary Crossentropy  
    - Optimizer: Adam (learning rate = 0.001)  
    - Batch Size: 32  
    - Epochs: 15 with **early stopping** (patience = 3)  

---

## 📊 Dataset Information
- **IMDB Dataset:** 50,000 reviews available on Kaggle.  
    - [Dataset Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?datasetId=134715&sortBy=dateRun&tab=profile)
- **Metacritic Dataset:** 151 manually scraped reviews used for further evaluation.

---

## 🧠 Model Architecture
### 🔹 Embedding Layer
- **Input Dimension:** 20,000  
- **Output Dimension:** 128  
- **Input Length:** 400  

### 🔹 Recurrent Layer
- **Type:** SimpleRNN  
- **Units:** 64  
- **Activation:** Tanh  
- **Dropout:** 0.2  

### 🔹 Fully Connected Layer
- **Neurons:** 1  
- **Activation:** Sigmoid (for binary classification)  

---

## 🏋️ Training Details
- **Dataset Split:** 80% training, 20% testing from 40,000 sampled IMDB reviews.
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam (learning rate = 0.001)  
- **Batch Size:** 32  
- **Epochs:** 15 (with early stopping after 3 epochs if validation loss did not improve).  

---

## 📈 Results & Observations
- **Training Accuracy:** Achieved ~88% accuracy after 10 epochs.  
- **Validation Accuracy:** Stabilized around ~85%, indicating good generalization.  
- **Test Accuracy on Metacritic:** Achieved ~91%, suggesting that the model performed well across datasets.  
- **Overfitting Mitigation:** Used dropout and early stopping to prevent overfitting.  
- **Limitations:**  
    - Lower performance on Metacritic reviews suggests that a more complex model, such as **LSTM or GRU**, may perform better.  

---

## 🚀 Managerial Insights & Business Applications
- **Customer Sentiment Monitoring:**  
    - Analyze customer opinions about movies, products, or services for better insights.
- **Brand Reputation Analysis:**  
    - Track sentiment trends to manage brand reputation and prevent PR crises.
- **Automated Review Filtering:**  
    - Identify and filter fake or irrelevant reviews automatically.

---

## 🔮 Future Improvements
- **Use LSTM/GRU Models:**  
    - RNNs have limited memory, which can be improved by switching to LSTM or GRU models for capturing long-term dependencies.
- **Advanced Preprocessing:**  
    - Implement lemmatization, n-grams, and stop-word removal to improve data quality.
- **Dataset Expansion:**  
    - Combine IMDB and Metacritic datasets to improve model robustness.
- **Real-time Sentiment Tracking:**  
    - Develop a real-time dashboard to monitor sentiment trends and provide actionable insights.

