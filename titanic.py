import pandas as pd

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_ids =test_data["PassengerId"]

def clean(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
    
    cols = ["SibSp", "Parch", "Fare", "Age"]
    data[cols] = data[cols].fillna(data[cols].median())
    

    data['Embarked'] = data['Embarked'].fillna("U")
    return data

train_data = clean(train_data)
test_data = clean(test_data)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cols = ["Sex", "Embarked"]

for col in cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])  
print(le.classes_)
    
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

y=train_data["Survived"]
x=train_data.drop("Survived", axis=1)
x_train, x_val, y_train, y_val=train_test_split(x,y, test_size=0.2, random_state=42)

clf=LogisticRegression(random_state=0, max_iter=1000).fit(x_train, y_train)
predictions=clf.predict(x_val)
from sklearn.metrics import accuracy_score
accuracy_score(y_val, predictions)

submission_preds= clf.predict(test_data)



submission = pd.DataFrame({
    "PassengerId": test_ids.values,  
    "Survived": submission_preds
})

submission.to_csv("submission.csv",index=False)
print("Prediction and submission file created successfully.")

