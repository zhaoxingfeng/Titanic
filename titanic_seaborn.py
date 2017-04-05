# coding:utf-8
"""
作者：zhaoxingfeng	日期：2017.04.05
版本：1.0
功能：kaggle泰坦尼克号各种分析，seaborn库学习
参考URL：
http://nbviewer.jupyter.org/github/jmportilla/Udemy-notes/blob/master/Intro%20to%20Data%20Projects%20-%20Titanic.ipynb
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('chained_assignment', None)  # 不显示DataFrame替换warn提示

titanic_df = pd.read_csv('titanic_train.csv')
# print titanic_df.info()

# 性别分布
sns.factorplot(x='Sex', data=titanic_df, kind="count", size=4, aspect=2)


# 船舱等级-性别分布
# sns.factorplot(x='Pclass', data=titanic_df, hue='Sex', kind="count")

# 判断男、女、小孩
def male_famle_child(passenger):
    age, sex = passenger
    if age < 16:
        return "Child"
    else:
        return sex
# 新增Person字段，存储男、女、小孩
titanic_df["Person"] = titanic_df[["Age", "Sex"]].apply(male_famle_child, axis=1)

# 3个船舱等级中男、女、小孩的数量
# sns.factorplot(x="Pclass", data=titanic_df, hue="Person", kind="count")

# 将年龄段间距分为20段，查看各年龄段人数发布
# titanic_df['Age'].hist(bins=20)

# Person字段的数量统计
titanic_df["Person"].value_counts()

# Person字段年龄分布密度
def person_age_kde():
    fig = sns.FacetGrid(titanic_df, hue="Person", aspect=4)
    # 使用map函数映射kde，以Age作为X轴，shade为是否填充线下阴影
    fig.map(sns.kdeplot, "Age", shade=True)
    oldest = titanic_df["Age"].max()
    fig.set(xlim=(0, oldest))
    fig.add_legend()
# person_age_kde()

# 三种船舱等级年龄分布密度
def pclass_age_kde():
    fig = sns.FacetGrid(titanic_df, hue="Pclass", aspect=4)
    fig.map(sns.kdeplot, 'Age', shade=True)
    oldest = titanic_df['Age'].max()
    fig.set(xlim=(0, oldest))
    fig.add_legend()
# pclass_age_kde()

# 不同船舱号的人数
def Cabin_hist():
    deck = titanic_df["Cabin"].dropna()
    levels = []
    for level in deck:
        levels.append(level[0])
    cabin_df = pd.DataFrame(levels, columns=["Cabin"])
    # 因为T船舱的数量太小，故删除
    cabin_df = cabin_df[cabin_df.Cabin != "T"]
    # palette：summer,winter_d,Blues,Set2
    sns.factorplot("Cabin", data=cabin_df, palette="winter_d", kind="count")
# Cabin_hist()

# 统计登船港口的数量分布
def Embarked_hist():
    sns.factorplot("Embarked", data=titanic_df, hue="Pclass", x_order=["C", "Q", "S"], kind="count")
# Embarked_hist()

# 统计单身及有家庭的人数分布，大于1表示有兄弟姐妹或者父母孩子
def Alone_hist():
    titanic_df["Alone"] = titanic_df.SibSp + titanic_df.Parch
    # 修改Alone字段的数字为Alone或者with family
    titanic_df["Alone"].loc[titanic_df["Alone"] > 0] = "With Family"
    titanic_df["Alone"].loc[titanic_df["Alone"] == 0] = "Alone"
    sns.factorplot(x="Alone", data=titanic_df, hue="Pclass", palette="Blues", kind="count")
# Alone_hist()

# 统计存活的以及没存活的人数分布
def yesno_survive():
    # 将Survivor字段的0,1映射成no与yes
    titanic_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})
    sns.factorplot("Survivor", data=titanic_df, palette="Set2", kind="count")
    # 不同船舱等级生还概率
    sns.factorplot(x="Pclass", y="Survived", data=titanic_df, x_order=[1, 2, 3])
    # 不同船舱等级、不同性别生还概率
    sns.factorplot(x="Pclass", y="Survived", hue='Person', data=titanic_df, x_order=[1, 2, 3])
# yesno_survive()

# age和生还的关系
def age_survive():
    # age
    sns.lmplot(x='Age', y='Survived', data=titanic_df)
    # age - pclass
    sns.lmplot(x='Age', y='Survived', hue='Pclass', data=titanic_df, palette='winter')
    # x_bins
    sns.lmplot(x='Age', y='Survived', hue='Pclass', data=titanic_df,
               palette='winter', x_bins=[10, 20, 40, 60, 80])
    # age - sex
    sns.lmplot(x='Age', y='Survived', hue='Sex', data=titanic_df,
               palette='winter', x_bins=[10, 20, 40, 60, 80])
# age_survive()

plt.savefig('titanic_seaborn.png', dpi=600)
plt.show()
