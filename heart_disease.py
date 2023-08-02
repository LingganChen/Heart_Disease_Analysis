#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Analysis

# # Introduction
# Heart disease, also known as cardiovascular disease, is a leading cause of morbidity and mortality worldwide. It encompasses a range of conditions affecting the heart and blood vessels, including coronary artery disease, heart failure, arrhythmias, and valvular heart diseases. Heart disease affects people of all ages and backgrounds, and its prevalence has been steadily increasing due to various risk factors, including sedentary lifestyles, poor dietary habits, smoking, and stress.
# 
# Understanding the underlying factors and risk predictors associated with heart disease is crucial for early detection, prevention, and effective management. Analyzing and interpreting relevant data can provide valuable insights into the prevalence, patterns, and potential interventions to combat this significant public health concern.

# # Data Source
# The dataset used in this article is the Cleveland Heart Disease dataset taken from the UCI repository.
# Link: http://archive.ics.uci.edu/dataset/45/heart+disease
# 
# 

# # Feature Description:
# 1. **age** : age in years
# 2. **sex** : (1 = male; 0 = female)
# 3. **cp** : chest pain type
#    - Value 1: typical angina
#    - Value 2: atypical angina
#    - Value 3: non-anginal pain
#    - Value 4: asymptomatic
# 
# 4. **trestbps**: displays the resting blood pressure value of an individual in mmHg (unit)
# 5. **chol** : serum cholestoral
# 6. **fbs** : fasting blood sugar > 120 mg/dl  (1 = true; 0 = false)
# 7. **restecg**: resting electrocardiographic results
# 
#       - Value 0: normal
#       - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#       - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# 8. **thalach**: maximum heart rate achieved
# 9. **exang** : exercise induced angina (1 = yes; 0 = no)
# 10. **oldpeak** : ST depression induced by exercise relative to rest
# 11. **slope** : the slope of the peak exercise ST segment
#        - Value 1: upsloping
#        - Value 2: flat
#        - Value 3: downsloping
# 12. **ca** : number of major vessels (0-3) colored by flourosopy
# 13. **thal** : 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. **presence** : the presence of heart disease in the patient, 0 = absence, 1 = present

# In[84]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from warnings import simplefilter


# In[85]:


# reading csv files
df =  pd.read_csv('processed.cleveland.data',header = None)
df.columns = ["age", "sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","presence"]
df


# # Check missing values

# There are no missing values in the dataframe, however, at row 302, there is a "?" as value from column ca, and the Dtype for column ca is an object, we assume that "?" is a replacement for a missing value, same goes for columm thal. We can replace the missing values by replacing it with the mean of the column.

# In[86]:


df.info()


# In[87]:


# reading csv files
df['thal'] = df['thal'].replace('?', np.nan).astype(float)
df['thal'] = df['thal'].fillna(df['thal'].mean()).round(1)
df['ca'] = df['ca'].replace('?', np.nan).astype(float)
df['ca'] = df['ca'].fillna(df['ca'].mean()).round(1)


# Now the missing values are placement with the mean of the column.

# In[88]:


df.info()


# # Correlation Heatmap

# In[89]:


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# # Age and Gender Distribution

# Based on the plot, it is evident that individuals with heart disease tend to be predominantly aged above 50. However, we found no discernible correlation between gender and the presence of heart disease.

# In[90]:


plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='age', bins=20, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Box plot to compare the distribution of age for different heart disease presence levels
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='presence', y='age', palette='coolwarm')
plt.title('Box Plot: Age Distribution across Heart Disease Presence Levels')
plt.xlabel('Heart Disease Presence')
plt.ylabel('Age')
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['0', '1', '2', '3', '4'])
plt.show()


# In[91]:


# barplot of age vs sex
df['sex'] = df['sex'].map({0: 'female', 1: 'male'})
sns.catplot(kind='bar', data=df, y='age', x='sex', hue='presence')
plt.title('Distribution of age vs sex with the target class')
plt.show()


# # Chest Pain Types vs Heart Disease Presence

# Based on the visualization of the distribution of chest pain types and heart disease presence, we observe the following:
# 
# - A significant proportion of patients (most of them) have chest pain type 4 (asymptomatic), indicating no chest pain.
# - Similarly, the majority of patients exhibit heart disease presence level 0, indicating no indication of heart disease.
# - However, it is essential to note that the absence of heart disease presence does not necessarily correlate with the absence of chest pain. There are cases where patients with chest pain type 4 (asymptomatic) have a higher number of instances with heart disease presence compared to those without heart disease presence.
# - Conversely, a large portion of patients with chest pain type 3 (non-anginal pain) has no heart disease presence.
# 
# In summary, the relationship between chest pain types and heart disease presence is not straightforward, and other factors may influence their correlation. Further analysis is required to understand the underlying patterns and potential risk factors associated with heart disease in these patients.

# In[92]:


cp_counts = df['cp'].value_counts()

# Create a bar chart for the distribution of chest pain types
plt.figure(figsize=(8, 6))
plt.bar(cp_counts.index, cp_counts.values)
plt.xlabel('Chest Pain Types (cp)')
plt.ylabel('Count')
plt.title('Distribution of Chest Pain Types')
plt.xticks(cp_counts.index)
plt.show()

# Count the occurrences of heart disease presence (presence)
presence_counts = df['presence'].value_counts()

# Create a bar chart for the distribution of heart disease presence
plt.figure(figsize=(8, 6))
plt.bar(presence_counts.index, presence_counts.values)
plt.xlabel('Heart Disease Presence')
plt.ylabel('Count')
plt.title('Distribution of Heart Disease Presence')
plt.xticks(presence_counts.index)
plt.show()


# In[93]:


cp_labels = {
    1: 'Typical Angina',
    2: 'Atypical Angina',
    3: 'Non-Anginal Pain',
    4: 'Asymptomatic'
}
df['cp_label'] = df['cp'].map(cp_labels)

# Create a bar chart to show the distribution of chest pain types
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='cp_label')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.title('Distribution of Chest Pain Types')
plt.xticks(rotation=45, ha='right')
plt.show()

# Create a stacked bar chart to visualize the relationship between chest pain types and heart disease presence
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='cp_label', hue='presence')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.title('Chest Pain Type Analysis by Heart Disease Presence')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Presence', labels=['No Heart Disease', 'Presence 1', 'Presence 2', 'Presence 3', 'Presence 4'])
plt.show()


# # Thalassemia (thal) Analysis
# Patients with a thal value of 7.0 are more prone to having heart disease, while the majority of patients exhibit a thal value of 3.0, which is considered normal. Among those patients with a thal value of 3.0, the majority do not have heart disease.

# In[94]:


# Create a bar plot to visualize the distribution of heart disease presence for each thal value
plt.figure(figsize=(8, 6))
sns.countplot(x='thal', hue='presence', data=df, palette='Set1')
plt.xlabel('Thalassemia Value')
plt.ylabel('Count')
plt.title('Distribution of Heart Disease Presence for each Thalassemia Value')
plt.legend(title='Heart Disease Presence', labels=['0', '1', '2', '3', '4'])
plt.show()


# # Resting Blood Pressure Analysis

# In[95]:


# Create a histogram to show the distribution of resting blood pressure
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='trestbps', bins=20, kde=True)
plt.xlabel('Resting Blood Pressure (mmHg)')
plt.ylabel('Count')
plt.title('Distribution of Resting Blood Pressure')
plt.show()

# Create a box plot to visualize the relationship between resting blood pressure and heart disease presence
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='presence', y='trestbps')
plt.xlabel('Presence of Heart Disease Level')
plt.ylabel('Resting Blood Pressure (mmHg)')
plt.title('Resting Blood Pressure Analysis by Heart Disease Presence')
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['0', '1', '2', '3', '4'])
plt.show()


# # Max Heart Rate Analysis

# In[96]:


# Scatter plot to visualize the relationship between thalach and heart disease presence
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='thalach', y='presence', hue='presence', palette='coolwarm')
plt.title('Scatter Plot: Maximum Heart Rate (thalach) vs. Heart Disease Presence')
plt.xlabel('Maximum Heart Rate (thalach)')
plt.ylabel('Heart Disease Presence')
plt.show()

# Box plot to compare the distribution of thalach for different heart disease presence levels
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='presence', y='thalach', palette='coolwarm')
plt.title('Box Plot: Maximum Heart Rate (thalach) across Heart Disease Presence Levels')
plt.xlabel('Heart Disease Presence')
plt.ylabel('Maximum Heart Rate (thalach)')
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['0', '1', '2', '3', '4'])
plt.show()


# # Data Preprocessing

# In[106]:


# reading csv files
df =  pd.read_csv('processed.cleveland.data',header = None)
df.columns = ["age", "sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","presence"]


# In[107]:


df['thal'] = df['thal'].replace('?', np.nan).astype(float)
df['thal'] = df['thal'].fillna(df['thal'].mean()).round(1)
df['ca'] = df['ca'].replace('?', np.nan).astype(float)
df['ca'] = df['ca'].fillna(df['ca'].mean()).round(1)


# In[108]:


#we made all presence level 1-4 to value 1 to indicate that there is a heart disease
df['presence'] = df.presence.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})


# In[110]:


#set features = X and target = y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[111]:


#split the data, 80% training and 20% testing 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # SVM

# In[117]:


from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize SVM classifier with a smaller C value
classifier = SVC(kernel='rbf', C=0.1)

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform feature selection using SelectKBest
selector = SelectKBest(chi2, k=13)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Perform k-fold cross-validation on the selected features
cv_accuracy = cross_val_score(classifier, X_train_selected, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy: {:.2f}%".format(cv_accuracy.mean() * 100))

# Fit the model on the training data with the selected features
classifier.fit(X_train_selected, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_selected)

# Confusion matrix for the test set
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm_test = confusion_matrix(y_pred, y_test)

# Calculate accuracy for training and test sets
y_pred_train = classifier.predict(X_train_selected)
cm_train = confusion_matrix(y_pred_train, y_train)
print()
print('Accuracy for training set for SVM = {:.2f}%'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train) * 100))
print('Accuracy for test set for SVM = {:.2f}%'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test) * 100))

# Print other metrics
print('\nClassification Report for Test Set:')
print(classification_report(y_test, y_pred))


# # Logistic Regression

# In[120]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

# Initialize the Logistic Regression classifier with L2 regularization
# Set C to a smaller value (e.g., 0.1) to increase regularization strength
classifier = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.1)

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform k-fold cross-validation on the scaled features
cv_accuracy = cross_val_score(classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy: {:.2f}%".format(cv_accuracy.mean() * 100))

# Fit the model on the scaled training data
classifier.fit(X_train_scaled, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_scaled)

# Confusion matrix for the test set
cm_test = confusion_matrix(y_pred, y_test)

# Calculate accuracy for training and test sets
y_pred_train = classifier.predict(X_train_scaled)
cm_train = confusion_matrix(y_pred_train, y_train)
print()
print('Accuracy for training set for Logistic Regression = {:.2f}%'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train) * 100))
print('Accuracy for test set for Logistic Regression = {:.2f}%'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test) * 100))
# Print other metrics
print('\nClassification Report for Test Set:')
print(classification_report(y_test, y_pred))


# # Naive Bayes

# In[121]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2

# Initialize the Naive Bayes classifier
classifier = GaussianNB()

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform feature selection using SelectKBest
selector = SelectKBest(chi2, k=9)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Perform k-fold cross-validation on the selected features
cv_accuracy = cross_val_score(classifier, X_train_selected, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy: {:.2f}%".format(cv_accuracy.mean() * 100))

# Fit the model on the scaled and selected training data
classifier.fit(X_train_selected, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_selected)

# Confusion matrix for the test set
cm_test = confusion_matrix(y_pred, y_test)

# Calculate accuracy for training and test sets
y_pred_train = classifier.predict(X_train_selected)
cm_train = confusion_matrix(y_pred_train, y_train)
print()
print('Accuracy for training set for Naive Bayes = {:.2f}%'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train) * 100))
print('Accuracy for test set for Naive Bayes = {:.2f}%'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test) * 100))
# Print other metrics
print('\nClassification Report for Test Set:')
print(classification_report(y_test, y_pred))


# # Decision Tree

# In[122]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# Initialize the Decision Tree classifier with constraints
classifier = DecisionTreeClassifier(max_depth=3, min_samples_split=5)

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform feature selection using SelectKBest
selector = SelectKBest(chi2, k=9)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Perform k-fold cross-validation on the selected features
cv_accuracy = cross_val_score(classifier, X_train_selected, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy: {:.2f}%".format(cv_accuracy.mean() * 100))

# Fit the model on the scaled and selected training data
classifier.fit(X_train_selected, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_selected)

# Confusion matrix for the test set
cm_test = confusion_matrix(y_pred, y_test)

# Calculate accuracy for training and test sets
y_pred_train = classifier.predict(X_train_selected)
cm_train = confusion_matrix(y_pred_train, y_train)
print()
print('Accuracy for training set for Decision Tree = {:.2f}%'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train) * 100))
print('Accuracy for test set for Decision Tree = {:.2f}%'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test) * 100))
# Print other metrics
print('\nClassification Report for Test Set:')
print(classification_report(y_test, y_pred))


# # Random Forest

# In[123]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

# Initialize the Random Forest classifier with more estimators and limited tree depth
classifier = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5)

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform k-fold cross-validation on the Random Forest classifier
cv_accuracy = cross_val_score(classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy: {:.2f}%".format(cv_accuracy.mean() * 100))

# Fit the model on the scaled training data
classifier.fit(X_train_scaled, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_scaled)

# Confusion matrix for the test set
cm_test = confusion_matrix(y_pred, y_test)

# Calculate accuracy for training and test sets
y_pred_train = classifier.predict(X_train_scaled)
cm_train = confusion_matrix(y_pred_train, y_train)
print()
print('Accuracy for training set for Random Forest = {:.2f}%'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train) * 100))
print('Accuracy for test set for Random Forest = {:.2f}%'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test) * 100))
# Print other metrics
print('\nClassification Report for Test Set:')
print(classification_report(y_test, y_pred))

