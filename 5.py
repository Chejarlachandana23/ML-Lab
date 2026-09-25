import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred,target_names=iris.target_names)

print(f"Accuracy:{accuracy:.2f}")
print("\nClassification Report:")
print(report)
