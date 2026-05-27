#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[5]:


df = pd.read_csv("Iris (1).csv")


# In[7]:


print(df.head())
print(df.info())
print(df['Species'].value_counts())


# In[22]:


X = df.drop('Species', axis=1)
y = df['Species']


# In[24]:


print(X)


# In[26]:


print(y)


# In[28]:


le = LabelEncoder()
y = le.fit_transform(y)


# In[30]:


print(y)


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# In[34]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# In[36]:


lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[38]:


nb = GaussianNB()
nb.fit(X_train, y_train)


# In[58]:


model = knn
knn_pred = model.predict(X_test)


# In[60]:


model = lr
lr_pred = model.predict(X_test)


# In[62]:


model = nb
nb_pred = model.predict(X_test)


# In[64]:


print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))


# In[66]:


print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))


# In[68]:


print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))


# In[70]:


print(confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred))


# In[72]:


# KNN Accuracy
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

# Logistic Regression Accuracy
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Naive Bayes Accuracy
nb_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

# Compare Results
comparison = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes'],
    'Accuracy': [knn_acc, lr_acc, nb_acc]
})

print(comparison)


# In[74]:


import time
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dictionary to store results
results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": [],
    "Execution Time": []
}

# ---------------- KNN ----------------
start = time.time()

knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

end = time.time()

results["Model"].append("KNN")
results["Accuracy"].append(accuracy_score(y_test, knn_pred))
results["Precision"].append(precision_score(y_test, knn_pred, average='weighted'))
results["Recall"].append(recall_score(y_test, knn_pred, average='weighted'))
results["F1-Score"].append(f1_score(y_test, knn_pred, average='weighted'))
results["Execution Time"].append(end - start)

# ---------------- Logistic Regression ----------------
start = time.time()

lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

end = time.time()

results["Model"].append("Logistic Regression")
results["Accuracy"].append(accuracy_score(y_test, lr_pred))
results["Precision"].append(precision_score(y_test, lr_pred, average='weighted'))
results["Recall"].append(recall_score(y_test, lr_pred, average='weighted'))
results["F1-Score"].append(f1_score(y_test, lr_pred, average='weighted'))
results["Execution Time"].append(end - start)

# ---------------- Naive Bayes ----------------
start = time.time()

nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

end = time.time()

results["Model"].append("Naive Bayes")
results["Accuracy"].append(accuracy_score(y_test, nb_pred))
results["Precision"].append(precision_score(y_test, nb_pred, average='weighted'))
results["Recall"].append(recall_score(y_test, nb_pred, average='weighted'))
results["F1-Score"].append(f1_score(y_test, nb_pred, average='weighted'))
results["Execution Time"].append(end - start)

# Create comparison table
comparison_df = pd.DataFrame(results)

# Display results
print(comparison_df)


# In[ ]:




