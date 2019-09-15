import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 选取有效特征
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']

X_train = train[selected_features]
X_test = train[selected_features]
y_train = train['Survived']

# 补全空缺值
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)

X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

# 特征向量化
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.transform(X_test.to_dict(orient='record'))


rfc = RandomForestClassifier()
sgbc = XGBClassifier()

rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerID':test['PassengerId'], 'Survived':rfc_y_predict})
rfc_submission.to_csv('rfc_submission.csv', index=False)

xgbc.fit(X_train, y_train)
xgbc_y_predict = xgbc.predict(X_test)
xgbc_submission = pd.DataFrame({'PassengerID':test['PassengerId'], 'Survived':xgbc_y_predict})
xgbc_submission.to_csv('xgbc_submission.csv', index=False)
params = {'max_depth':range(2,7), 'n_estimators':range(100, 1100, 200), 'learning_rate':[0.05, 0.1, 0.25, 0.5, 1.0]}

xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
gs.fit(X_train, y_train)


xgbc_best_y_predict = gs.predict(X_test)
xgbc_best_submission = pd.DataFrame({'PassengerID':test['PassengerId'], 'Survived':xgbc_best_y_predict})
xgbc_best_submission.to_csv('xgbc_best_submission.csv', index=False)
