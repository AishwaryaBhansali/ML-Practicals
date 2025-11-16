import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetes.csv')
print(df.head())
print(df.columns)
imputer = SimpleImputer(missing_values=0, strategy='mean', copy=False)

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']

for col in feature_cols:
    print(f"Missing cols : {len(df.loc[df[col] == 0])}")

df[feature_cols] = imputer.fit_transform(df[feature_cols])

for col in feature_cols:
    print(f"Missing cols : {len(df.loc[df[col] == 0])}")

#Scaling
scaler = StandardScaler()
x = df[feature_cols]
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


models = {
    'Bagging' : BaggingClassifier(estimator = DecisionTreeClassifier(), n_estimators=50, random_state=42),
    'RandomForestClassifier' : RandomForestClassifier(n_estimators=50, random_state=42),
    'AdaBoostClassifier' : AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42, learning_rate = 1),
    'GradientBoostingClassifier' : GradientBoostingClassifier(n_estimators=50, random_state=42, learning_rate=0.1)
}

accuracy_result = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(preds, y_test)
    accuracy_result[name] = acc
    print(f"{name} : {acc}")

plt.figure(figsize=(12,5))
plt.bar(accuracy_result.keys(), accuracy_result.values(), color=['red', 'blue', 'green', 'orange'])
plt.title("Comparsion of various ensemble algorithms.")
for i,v in enumerate(accuracy_result.values()):
    plt.text(i, v+0.01, f"{v : .2f}")
plt.show()