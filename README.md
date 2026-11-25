# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8. Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: ROUNAK SINGH
RegisterNumber:  212225240125
*/
```
```py
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result
```

```py
import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')
```

```py
data.info()
```

```py
data.isnull().sum()
```

```py
x=data["v1"].values
y=data["v2"].values
```

```py
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
```

```py
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```

```py
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```


```py
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
```

```py
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:

<img width="876" height="190" alt="image" src="https://github.com/user-attachments/assets/0d15f57c-b2a7-4ec4-bad5-350b6a9d5e2f" />

<img width="902" height="290" alt="image" src="https://github.com/user-attachments/assets/52d60473-12b7-4775-bd82-2d8ca6771d20" />

<img width="507" height="330" alt="image" src="https://github.com/user-attachments/assets/14d269c4-8c2d-40f1-a596-7c21dd3b9e19" />

<img width="270" height="206" alt="image" src="https://github.com/user-attachments/assets/7e1b62b3-6e87-4b71-a5ef-9eaaabcf7e3e" />

<img width="805" height="225" alt="image" src="https://github.com/user-attachments/assets/538dfeab-b877-49c5-ae7b-4f1bb1580205" />

<img width="518" height="147" alt="image" src="https://github.com/user-attachments/assets/9c563559-0bc8-4546-9911-0490eae63cd0" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
