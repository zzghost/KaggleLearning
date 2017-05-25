#-*- encoding:utf-8 -*-
'''
Classic dataset on Kaggle:Titanic.
'''
import pandas as pd
import numpy as np

#Step1:know data
data_train = pd.read_csv("./data/train.csv")
#As we can see, there are missing values in the dataset.Like age:714, cabin:204
#print(data_train)
#print(data_train.info())
#print(data_train.describe())


#Step2:analyse data
#step2.1: watch the distribution of each attributes
import matplotlib.pyplot as plt
'''
step2.1:watch the distribution of each attributes
fig = plt.figure()
fig.set(alpha=0.2)
plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title("Survived(1:survived)")
plt.ylabel("Numbers")

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.ylabel("Classes of passengers")
plt.ylabel("Numbers")

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("Age")
plt.grid(b=True, which='major', axis='y')
plt.title('Age(1:Surved')

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")
plt.ylabel("Density")
plt.title("The distribution of different class passengers")
plt.legend(('first', 'second', 'third'), loc='best')

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("Each embarked")
plt.ylabel("numbers")
plt.show()
'''
#step2.2: watch the relation between attributes and results.
'''
atrribute1: class
fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({"survived": Survived_1, "unsurvived": Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("Each class survived status")
plt.xlabel("Classes")
plt.ylabel("Numbers")
plt.show()
'''
'''
attribute2: sex
fig = plt.figure()
fig.set(alpha=0.2)
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({"male": Survived_m, "female": Survived_f})
df.plot(kind='bar', stacked=True)
plt.title("Sex vs Survived")
plt.xlabel("Sex")
plt.ylabel("Numbers")
plt.show()
'''
'''
attribute3: sex and class

fig = plt.figure()
fig.set(alpha=0.65)
plt.title("Sex and class")

ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='Female high class', color='#FA2419')
ax1.set_xticklabels(['survived', 'unsurvived'], rotation=0)
ax1.legend(['female, high class'], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='Female low class', color='pink')
ax2.set_xticklabels(['survived', 'unsurvived'], rotation=0)
ax2.legend(['female, low class'], loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='Male high class', color='lightblue')
ax3.set_xticklabels(['survived', 'unsurvived'], rotation=0)
ax3.legend(['male, high class'], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='Male low class', color='steelblue')
ax4.set_xticklabels(['survived', 'unsurvived'], rotation=0)
ax4.legend(['male, low class'], loc='best')

plt.show()
'''

'''
attribute4: each embark

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'survived':Survived_1, 'unsurvived':Survived_0})
df.plot(kind='bar', stacked=True)


plt.show()
'''
'''
attribute5 : cabin

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({'yes':Survived_cabin, 'no':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.show()
'''

#step3: feature engineering
#first: dealing with the missing value;
#second: one-hot encoding
from sklearn.ensemble import RandomForestRegressor
def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    know_age = age_df[age_df.Age.notnull()].as_matrix()
    unknow_age = age_df[age_df.Age.isnull()].as_matrix()

    y = know_age[:, 0]
    x = know_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)

    predictedAges = rfr.predict(unknow_age[:, 1:])
    df.loc[ (df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr

def set_carbin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'no'
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_carbin_type(data_train)
#print(data_train)

dummies_cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
df = pd.concat([data_train, dummies_cabin, dummies_embarked, dummies_Sex, dummies_pclass], axis=1)
df.drop(['Pclass', 'Cabin', 'Embarked', 'Sex', 'Name', 'Ticket'], axis=1, inplace=True)

import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)

#logistic regression
from sklearn import linear_model
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

y = train_np[:, 0]
X = train_np[:, 1:]

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

#process the test data
data_test = pd.read_csv("./data/test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_carbin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)


test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("predict.csv", index=False)
