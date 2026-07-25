import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

data = pd.read_csv(
    "heart+disease/processed.cleveland.data",
    header=None,
    names=columns,
    na_values="?"
)

print(data.head())
print(data.shape)
print(data.isnull().sum())
data = data.dropna()

data["target"]=data["target"].apply(lambda x:0 if x==0 else 1)
X=data.drop("target",axis=1)
Y=data["target"]

X_train , X_test , Y_tarin,Y_test=train_test_split(
    X,Y ,test_size=0.2,random_state=42
)

model= DecisionTreeClassifier(max_depth=4,random_state=42)
model.fit(X_train,Y_tarin)

Y_pred=model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(Y_test,Y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

import pickle
with open("heart_model.pkl","wb") as file:
    pickle.dump(model,file)
print("Model saved successfully.")

sample_patient = [[
    52,   # age
    1,    # sex
    4,    # chest pain type
    125,  # resting blood pressure
    212,  # cholesterol
    0,    # fasting blood sugar
    1,    # resting ECG
    168,  # max heart rate
    0,    # exercise induced angina
    1.0,  # oldpeak
    2,    # slope
    0,    # ca
    3     # thal
]]

result = model.predict(sample_patient)

if result[0] == 0:
    print("Prediction: No Heart Disease")
else:
    print("Prediction: Heart Disease")