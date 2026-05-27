🌸 Iris Flower Classification using Supervised Machine Learning

📌 Project Overview

This project demonstrates Supervised Machine Learning using the famous Iris Flower Dataset.
The main goal is to classify iris flowers into different species using multiple machine learning algorithms and compare their performance.

The project uses:

✅ K-Nearest Neighbors (KNN)
✅ Logistic Regression
✅ Gaussian Naive Bayes
🌼 Dataset Information

The Iris dataset contains flower measurements for three species:

- 🌸 Iris-setosa
- 🌺 Iris-versicolor
- 🌷 Iris-virginica

Features Used
-----------------------------------------------------------------------
Feature	Description
-----------------------------------------------------------------------
SepalLengthCm	Length of sepal
--------------------------------------------------------------------
- SepalWidthCm	Width of sepal
- PetalLengthCm	Length of petal
- PetalWidthCm	Width of petal
----------------------------------------------------------------------
🛠️ Technologies & Libraries
----------------------------------------------------------------------
Tool / Library	Purpose
🐍 Python	Programming Language
📊 Pandas	Data Handling
🤖 scikit-learn	Machine Learning
🔢 NumPy	Numerical Operations
📂 Project Structure
├── Iris_Flower_Supervised_ML.py
├── Iris (1).csv
└── README.md
⚙️ Machine Learning Workflow
- 1️⃣ Import Libraries
The project imports essential libraries for:
Data preprocessing
Model building
Performance evaluation
- 2️⃣ Load Dataset
df = pd.read_csv("Iris (1).csv")
- 3️⃣ Data Preprocessing
Splitting features and target
Encoding species labels using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
- 4️⃣ Train-Test Split

Dataset is divided into:
---------------------------------------------------------------------------------------
📘 Training Set → 80%
📕 Testing Set → 20%
train_test_split(test_size=0.2, random_state=42)
- 5️⃣ Model Training

The following models are trained:
--------------------------------------------------------------------------------------------------
Model	Description
🔹 KNN	Classification based on nearest neighbors
🔹 Logistic Regression	Statistical classification model
🔹 Gaussian Naive Bayes	Probabilistic classifier
- 6️⃣ Model Evaluation

Evaluation metrics used:
------------------------------------------------------------------------------------------------
✅ Accuracy
✅ Precision
✅ Recall
✅ F1-Score
✅ Confusion Matrix
✅ Execution Time

📈 Algorithms Used
-------------------------------------------------------------------------------------------------
🔹 K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=3)
--------------------------------------------------------------
🔹 Logistic Regression
lr = LogisticRegression()
--------------------------------------------------------------
🔹 Gaussian Naive Bayes
nb = GaussianNB()
--------------------------------------------------------------
📊 Performance Comparison
---------------------------------------------------------------------------------------------
The project compares all models based on:
---------------------------------------------------------------------------------------------
- Metric	Description
- Accuracy	Correct predictions
- Precision	Positive prediction quality
- Recall	Ability to find positives
- F1-Score	Balance between precision & recall
- Execution Time	Training + prediction speed

▶️ How to Run the Project
-----------------------------------------------------------------------------------------------
- Step 1️⃣ Clone Repository
git clone 
- Step 2️⃣ Install Dependencies
pip install pandas scikit-learn
- Step 3️⃣ Run the Program
python Iris_Flower_Supervised_ML.py

📷 Sample Output
--------------------------------------------------------------------------------------------------
- KNN Accuracy: 1.0
- Logistic Regression Accuracy: 1.0
- Naive Bayes Accuracy: 1.0
🎯 Learning Outcomes

Through this project, one can learn:

📌 Data preprocessing
📌 Label encoding
📌 Train-test splitting
📌 Model training
📌 Classification evaluation
📌 Performance comparison
-----------------------------------------------------------------------------------------------------
🚀 Future Improvements
📈 Add visualization using Matplotlib/Seaborn
🤖 Add more ML algorithms
🌐 Deploy using Streamlit or Flask
📊 Hyperparameter tuning
