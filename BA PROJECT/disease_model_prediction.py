import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

#LOAD THE DATASET 

df=pd.read_csv('Training.csv/Training.csv')
print("\ndataset shape:",df.shape)
print("Available columns",df.columns.tolist)
print(df.head())

#Define Features and Targets 

label_col = "prognosis"
x= df.drop(columns=[label_col])
y= df[label_col]
print("\nNumber of Symptoms(Features):", x.shape[1])
print("Number of diseases(Classes):", len(y.unique()))

#Split Data into Training and Testing 

x_train, x_test , y_train , y_test  =train_test_split(
    x,y, test_size=0.2 , random_state=42, stratify=y
)

print("\n Training Set Shape:",x_train.shape)
print("Testing Set Shape:", x_test.shape)

#Train the Model  

print("\n Training the Model")
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=1,
    class_weight="balanced_subsample"
)
model.fit(x_train, y_train)
 #Evaluate the Model

y_pred = model.predict(x_test)

print("\n Model Evaluation")
print(" Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the Model

dump({
    "model": model,
    "features": list(x.columns),
    "classes": model.classes_
 }, "disease_prediction_model.joblib")











