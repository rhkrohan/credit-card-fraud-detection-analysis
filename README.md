# Credit Card Fraud Detection Analysis
## Introduction

This project focuses on developing a machine learning model to detect fraudulent credit card transactions using an Artificial Neural Network (ANN). The model is trained on a dataset of historical transaction data, where various features such as transaction amount, location, and customer demographics are used to predict the likelihood of fraud. Extensive data preprocessing, including one-hot encoding, feature scaling, and feature engineering (e.g., calculating geographic distance), ensures that the model effectively captures patterns associated with fraudulent activity. The resulting model can be integrated into real-time fraud detection systems, providing enhanced security and reducing financial losses for financial institutions.

### Key Features:
- **Data preprocessing and feature engineering** for optimal model performance.
- **ANN architecture** designed for binary classification of fraudulent vs. legitimate transactions.
- **Industrial relevance** with potential integration into real-time fraud prevention systems.

This project provides a comprehensive approach to tackling credit card fraud, leveraging the power of neural networks to improve detection accuracy.

Credit card fraud is a significant challenge in the financial industry, leading to substantial financial losses every year. By utilizing an Artificial Neural Network (ANN), this project aims to accurately predict fraudulent transactions based on historical transaction data. The effectiveness of the model is crucial for minimizing financial losses, improving security, and maintaining customer trust.

### Industrial Applications:

| Application                     | Description                                                                                                             |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| **Fraud Prevention Systems**     | The model can be integrated into real-time fraud detection systems used by banks, financial institutions, and payment processors. These systems monitor transactions as they occur and flag suspicious activities, allowing for immediate intervention before financial losses occur. |
| **Customer Protection**          | Implementing an effective fraud detection model helps protect customers from unauthorized transactions, thereby enhancing the overall customer experience and trust in financial services. |
| **Risk Management**              | Beyond immediate fraud detection, the insights gained from the model can inform broader risk management strategies, helping financial institutions better understand and mitigate various types of fraud. |

## Data Preprocessing

Data preprocessing is a critical step in preparing the dataset for training the machine learning model. This process ensures that the data is clean, structured, and suitable for input into the Artificial Neural Network (ANN). The dataset used in this project is sourced from Kaggle and contains detailed information on credit card transactions. The primary goal is to use this data to detect fraudulent transactions by training an Artificial Neural Network (ANN). The dataset is well-structured, with each observation representing a unique transaction, and it includes various features that describe transaction details, customer demographics, and merchant information.

### Dataset Overview

| **Attribute**              | **Description**                                                                                  |
|----------------------------|--------------------------------------------------------------------------------------------------|
| **Number of Observations**  | 14,446 transactions                                                                             |
| **Number of Features**      | 15 features                                                                                     |
| **Source**                  | Kaggle                                                                                          |
| **Feature Names**           | Description of each feature is provided below                                                   |
| **`trans_date_trans_time`** | Timestamp of the transaction.                                                                   |
| **`merchant`**              | Name of the merchant.                                                                           |
| **`category`**              | Category of the transaction.                                                                    |
| **`amt`**                   | Transaction amount.                                                                             |
| **`city`**                  | City where the transaction occurred.                                                            |
| **`state`**                 | State where the transaction occurred.                                                           |
| **`lat`**                   | Latitude of the customer's location.                                                            |
| **`long`**                  | Longitude of the customer's location.                                                           |
| **`city_pop`**              | Population of the city where the transaction occurred.                                          |
| **`job`**                   | Customer's job title.                                                                           |
| **`dob`**                   | Customer's date of birth.                                                                       |
| **`trans_num`**             | Unique transaction identifier.                                                                  |
| **`merch_lat`**             | Latitude of the merchant's location.                                                            |
| **`merch_long`**            | Longitude of the merchant's location.                                                           |
| **`is_fraud`**              | Indicator of whether the transaction is fraudulent (1 for fraud, 0 for non-fraud).              |

Below is a detailed explanation of the preprocessing steps performed in this project and the rationale behind each step.

## Data Preprocessing Summary

| **Step**                            | **Description**                                                                                                 | **Purpose**                                                                                          | **Method**                                                                                                                                  |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| **Loading the Dataset**             | The dataset is loaded into a pandas DataFrame for easy manipulation and analysis.                               | To structure the data for efficient preprocessing and analysis.                                       | `pd.read_csv(file_path)`                                                                                                                    |
| **Separating Features and Target**  | Independent variables (features) are separated from the dependent variable (target).                            | To distinguish between the data the model will learn from and the outcome it will predict.             | `X = data.drop(columns=['is_fraud']); y = data['is_fraud']`                                                                                 |
| **Cleaning the Target Variable**    | The `is_fraud` column is cleaned by extracting numeric values and converting them to integers.                  | To ensure the target variable is in a clean, numeric format suitable for training.                    | `y_cleaned = y.str.extract(r'(\d)').astype(int)`                                                                                            |
| **Converting Date Columns**         | Date columns (`trans_date_trans_time` and `dob`) are converted from strings to datetime objects.                | To enable extraction of time-related features, which are crucial for detecting temporal patterns.      | `pd.to_datetime()` with the appropriate date format.                                                                                        |
| **One-Hot Encoding of Categorical** | Categorical features (`merchant`, `category`, `city`, `job`, `state`) are converted into numerical format.      | To transform categorical data into a format suitable for machine learning models, particularly ANNs.   | `pd.get_dummies()`                                                                                                                          |
| **Feature Engineering**             | New features are derived from existing data, such as date, time, age, and distance between customer and merchant.| To enhance the model's predictive power by providing additional context and insights.                  | - **Extracting Date/Time:** `dt.date.apply(lambda x: x.toordinal())`; <br> - **Calculating Age:** `today - dob`; <br> - **Distance:** `geodesic` |
| **Scaling/Normalization**           | Feature scaling is applied to ensure all features have a mean of 0 and a standard deviation of 1.                | To prevent bias due to differing scales and to improve the model’s learning efficiency.               | `StandardScaler().fit_transform(X)`                                                                                                         |
| **Splitting the Data**              | The dataset is split into training and testing sets.                                                            | To evaluate the model's performance on unseen data and prevent overfitting.                           | `train_test_split(X_scaled, y, test_size=0.2, random_state=42)`                                                                             |

## Building the Artificial Neural Network (ANN)

The construction of the Artificial Neural Network (ANN) is a critical component of this project, as it directly impacts the model's ability to accurately predict fraudulent transactions. In this section, we delve into the architecture of the ANN, the rationale behind the design choices, and the techniques used to optimize the model's performance.

### ANN Architecture

The ANN developed for this project follows a Sequential model architecture, which is one of the most straightforward and commonly used methods for building neural networks. This model is constructed by stacking layers sequentially, where the output of one layer serves as the input to the next. The ANN in this project consists of the following layers:

1. **Input Layer**: 
   - The input layer receives the processed features from the dataset. The number of neurons in this layer corresponds to the number of features, which in this case is determined by the preprocessing steps.

2. **Hidden Layers**:
   
   | **Layer**          | **Number of Neurons** | **Activation Function** | **Rationale**                                                                                              |
   |--------------------|-----------------------|--------------------------|-------------------------------------------------------------------------------------------------------------|
   | **First Hidden Layer**  | 13                    | ReLU                      | The choice of 13 neurons was determined through experimentation. This number provides a balance between model complexity and performance, allowing the network to learn sufficient patterns without overfitting. |
   | **Second Hidden Layer** | 4                     | ReLU                      | The reduction to 4 neurons funnels the learned information, helping to distill the features and prevent overfitting as the network moves closer to the output layer. This structure encourages the network to generalize better to new, unseen data. |

3. **Output Layer**:
   - The output layer has a single neuron, as the task is a binary classification problem (fraud vs. non-fraud). The Sigmoid activation function is used in this layer, which is ideal for binary classification tasks. The Sigmoid function outputs a probability value between 0 and 1, which represents the model's prediction of whether a transaction is fraudulent or not.

### Activation Functions

Activation functions are crucial in a neural network as they introduce non-linearity, enabling the network to learn complex patterns in the data. For this ANN:

- **ReLU (Rectified Linear Unit)**: Used in the hidden layers, ReLU is the most widely used activation function in deep learning models. It is computationally efficient and helps prevent issues like the vanishing gradient problem, which can hamper the learning process in deeper networks.
  
- **Sigmoid**: Used in the output layer, the Sigmoid function is perfect for binary classification tasks. It converts the output into a probability score between 0 and 1, allowing for a smooth and interpretable prediction of either class (fraud or non-fraud).

### Tuning the Model

| **Component**            | **Details**                                                        | **Rationale**                                                                                                           |
|--------------------------|--------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| **Number of Neurons**     | 13 in the first hidden layer, 4 in the second hidden layer         | Experimentation led to these values as they offered a good balance between complexity and generalization, preventing overfitting. |
| **Activation Functions**  | ReLU in hidden layers, Sigmoid in output layer                     | ReLU efficiently handles non-linearity and avoids the vanishing gradient problem; Sigmoid provides smooth binary outputs, ideal for classification tasks. |
| **Optimizer**             | Adam Optimizer                                                     | Adam combines the best features of AdaGrad and RMSProp, offering efficient and adaptive learning rates, particularly useful for large datasets. |
| **Loss Function**         | Binary Cross-Entropy                                               | Ideal for binary classification, this function measures the divergence between predicted probabilities and actual labels, guiding the model to improve accuracy. |

### Model Training and Evaluation

During training, the model iteratively adjusts its weights using the Adam optimizer to minimize the binary cross-entropy loss. The use of ReLU in the hidden layers helps the model learn complex, non-linear relationships in the data, while the Sigmoid function in the output layer ensures that the predictions are probabilities, which are easy to interpret.

The architecture of this ANN, with its two hidden layers, has been optimized to balance model complexity and training performance, resulting in a network that is both powerful and efficient at detecting fraudulent transactions.
