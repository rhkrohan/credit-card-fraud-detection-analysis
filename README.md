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

---

### Preprocessing Workflow

```plaintext
+-----------------------------------+            +------------------------------------------+
|                                   |            |                                          |
|   Loading the Dataset             |            |  Separating Features and Target          |
|                                   |            |                                          |
+-----------------------------------+            +------------------------------------------+
           |                                     |      
           v                                     v
+-----------------------------------+    +------------------------------------------+
|                                   |    |                                          |
|   Cleaning the Target Variable    +--> |  Converting Date Columns                 |
|                                   |    |                                          |
+-----------------------------------+    +------------------------------------------+
           |                                     |
           v                                     v
+-----------------------------------+    +------------------------------------------+
|                                   |    |                                          |
|   One-Hot Encoding                +--> |  Feature Engineering                     |
|                                   |    |   - Extracting Date and Time Components  |
+-----------------------------------+    |   - Calculating Age                      |
           |                             |   - Calculating Distance                 |
           v                             +------------------------------------------+
+-----------------------------------+              |
|                                   |              v
|   Scaling/Normalization           +--> +------------------------------------------+
|                                   |    |                                          |
+-----------------------------------+    |  Splitting the Data                      |
           |                             |                                          |
           v                             +------------------------------------------+
+-----------------------------------+     
|                                   |     
|    Data Ready for Modeling        |     
|                                   |     
+-----------------------------------+     

