# Internship-Project

## Aim 
The goal of this mini project is to build a machine learning model that can classify iris flowers into three species: Setosa, Versicolour, and Virginica. This project guides you through the complete process of developing a machine learning model, from data exploration and preprocessing to model training, evaluation, and interpretation.

## Code
```
# Importing libraries for loading the iris dataset
import pandas as pd
from sklearn.datasets import load_iris
```

```
# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
```

```
# Import necessary libraries for training and testing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```

```
# Data exploration
print(data.head())
print(data.describe())
print(data['species'].value_counts())
```

![image](https://github.com/user-attachments/assets/d717b189-eff3-4c1b-b30e-3e50f5625511)

```
# Data preprocessing
X = data.drop('species', axis=1)
y = data['species']
```

```
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

```
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

```
# Model selection and training
model = LogisticRegression()
model.fit(X_train, y_train)
```

![image](https://github.com/user-attachments/assets/dcb89c4c-bdfa-42cd-83e5-a5a205c634c0)

```
# Model evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

![image](https://github.com/user-attachments/assets/fbe6a521-881c-40b3-b5c1-9dd2688bba29)

```
# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

![image](https://github.com/user-attachments/assets/24b15498-5f8a-43b5-aa32-12a0dcc06562)

```
# Import libraries for testing accuracy score
from sklearn.metrics import accuracy_score
```

```
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```


![image](https://github.com/user-attachments/assets/1f216c98-4d97-46d8-a8ae-428b89048506)

## Result
This mini project offers a comprehensive overview of the machine learning workflow, from data preprocessing to model evaluation and interpretation. It serves as a solid foundation for understanding the basic concepts and techniques required to develop and evaluate a classification model. For further enhancement, you could explore other classification algorithms, perform hyperparameter tuning, or apply cross-validation to ensure even more robust model performance.
