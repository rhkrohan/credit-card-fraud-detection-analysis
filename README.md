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
```
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




1. **Loading the Dataset**  
   The first step involves loading the dataset into a pandas DataFrame for easy manipulation and analysis. Working with the dataset in a DataFrame format allows for efficient application of data preprocessing techniques such as cleaning, transformation, and feature engineering.

2. **Separating Features (X) and Target (y)**  
   We separate the independent variables (features) from the dependent variable (target) to clearly define what the model will learn from (features) and what it will predict (target). This separation is crucial as it allows the model to learn patterns in the features that are predictive of the target variable.

3. **Cleaning the Target Variable (y)**  
   The target variable, `is_fraud`, initially contained mixed content, including numeric labels combined with timestamps. We cleaned this column by extracting only the numeric values and converting them into integers. Ensuring that the target variable contains only clean, numeric values is essential for accurate model training. The model needs to predict a binary outcome (fraud or not fraud), and any non-numeric content would hinder this process.

4. **Converting Date Columns to Datetime Format**  
   We converted the `trans_date_trans_time` and `dob` (date of birth) columns from string format to datetime objects. Converting these columns to datetime objects allows us to easily extract useful features such as the transaction date, transaction time, and customer age. These time-related features are often crucial in fraud detection, as fraudulent transactions may exhibit specific temporal patterns.

5. **One-Hot Encoding of Categorical Variables**  
   We applied one-hot encoding to convert categorical features such as `merchant`, `category`, `city`, `job`, and `state` into a numerical format suitable for model training. Machine learning models, particularly neural networks, require numerical input. One-hot encoding allows us to transform categorical variables into binary columns, where each category is represented as a separate feature with a value of 0 or 1.

6. **Feature Engineering**  
   We created new features from the existing data to enhance the model's predictive power. Feature engineering allows us to derive new insights from the data that may not be immediately apparent in the raw format. These engineered features can significantly improve the model's ability to detect fraud.

   a. **Extracting Date and Time Components**  
      We extracted the transaction date and time from the `trans_date_trans_time` column and transformed them into numerical features. Temporal features such as the date and time of a transaction can be highly predictive of fraudulent activity. By converting these into numerical values, the model can learn patterns related to when fraud is more likely to occur.

   b. **Calculating Age from Date of Birth**  
      We calculated the customer's age based on the `dob` column. Age is a demographic feature that may correlate with transaction behavior and the likelihood of fraud. By deriving age from the date of birth, we provide the model with additional context about the customer.

   c. **Calculating Distance Between Customer and Merchant**  
      We calculated the geographic distance between the customer's location and the merchant's location using their respective latitude and longitude coordinates. The distance between the customer and the merchant can be a strong indicator of fraudulent activity. Transactions occurring far from the customer's usual location might be suspicious and warrant closer scrutiny.

7. **Scaling/Normalization**  
   We applied feature scaling to ensure that all features contribute equally to the model during training. Features in the dataset can have different ranges, which can bias the model if not addressed. For example, transaction amounts might range from a few dollars to thousands, while age might only range from 18 to 80. Scaling ensures that each feature has a mean of 0 and a standard deviation of 1, allowing the model to learn more effectively.

8. **Splitting the Data into Training and Testing Sets**  
   Finally, we split the dataset into training and testing sets, with 80% of the data used for training and 20% used for testing. Splitting the data allows us to evaluate the model's performance on unseen data. The training set is used to fit the model, while the testing set provides an unbiased assessment of how well the model generalizes to new data.
