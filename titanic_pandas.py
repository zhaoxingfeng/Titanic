# coding:utf-8
"""
作者：zhaoxingfeng	日期：2017.04.02
版本：2.0
功能：随机森林建立kaggle泰坦尼克号乘客获救模型，pandas库学习
参考URL：http://www.cnblogs.com/voidleaf/p/6160459.html
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

"""
数据结构：PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
PassengerId：乘客ID；Survived：是否生还，0表示遇难，1表示生还；Pclass：阶级，1表示最高阶级，3最低；SibSp：同乘船的兄弟姐妹的数量；
Parch：是否有配偶同乘，1表示是；Ticket：船票编号；Fare：恐惧指数；Cabin：船舱号；Embarked：登船港口
"""

train_df = pd.read_csv('titanic_train.csv', header=0)

# female：0, Male：1
train_df['Gender'] = train_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Embarked缺失值采用众数填充
if len(train_df.Embarked[train_df.Embarked.isnull()]) > 0:
    train_df.loc[train_df.Embarked.isnull(), 'Embarked'] = train_df.Embarked.dropna().mode().values

# 建立Embarked字典，{'Q': 1, 'C': 0, 'S': 2}
Ports = list(enumerate(set(train_df['Embarked'])))
Ports_dict = {name: i for i, name in Ports}

train_df.Embarked = train_df.Embarked.map(lambda x: Ports_dict[x]).astype(int)

# Age缺失值采用中位数填充
if len(train_df.Age[train_df.Age.isnull()]) > 0:
    train_df.loc[train_df.Age.isnull(), 'Age'] = train_df.Age.dropna().median()

# 剔除不起作用的列
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

test_df = pd.read_csv('titanic_test.csv', header=0)
test_df['Gender'] = test_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Embarked缺失值采用众数填充，并转化为数字
if len(test_df.Embarked[test_df.Embarked.isnull()]) > 0:
    test_df.Embarked[test_df.Embarked.isnull()] = test_df.Embarked.dropna().mode().values
test_df.Embarked = test_df.Embarked.map(lambda x: Ports_dict[x]).astype(int)

# Age缺失值采用中位数填充
if len(test_df.Age[test_df.Age.isnull()]) > 0:
    test_df.loc[(test_df.Age.isnull()), 'Age'] = test_df['Age'].dropna().median()

# Fare缺失值采用中位数填充
if len(test_df.Fare[test_df.Fare.isnull()]) > 0:
    median_fare = np.zeros(3)
    for f in range(0, 3):
        median_fare[f] = test_df[test_df.Pclass == f+1]['Fare'].dropna().median()
    for f in range(0, 3):
        test_df.loc[(test_df.Fare.isnull()) & (test_df.Pclass == f+1), 'Fare'] = median_fare[f]

# 收集PassengerId
ids = test_df['PassengerId'].values
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

print 'Training...'
train_data = train_df.values
test_data = test_df.values
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

print 'Predicting...'
output = forest.predict(test_data).astype(int)

temp = pd.concat([pd.DataFrame(ids, columns=["PassengerId"]),
                  pd.DataFrame(output, columns=["Survived"])], axis=1)
temp.to_csv('titanic_pandas.csv', header=['ID', 'Survive'], sep=',', index=False, mode='w', encoding=None)
print 'Done.'
