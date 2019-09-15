# Titanic: Machine Learning from Disaster

[TOC]

## 0 整体流程

1. 了解问题 & 获取训练集和测试集
2. 整理、准备、清洗数据
3. 分析、识别模式并探索数据
4. 建模、预测和解决问题
5. 可视化、报告和呈现问题解决步骤和最终解决方案



**数据处理流程：**

- **Classifying 分类**
  - 对样本进行分类；了解不同类与解决方案目标的含义或相关性
- **Correlating 特征相关性**
  - 了解数据集中的哪些特征对解决方案目标有重大贡献；了解特征之间的相关性；关联某些特征可能有助于纠正特征或创建新特征

- **Converting 转换**
  - 将所有需要的特征转换成数字形式，便于模型学习
- **Completing 数据补全**
  - 将缺失值补全，完整的数据将有助于模型的学习

- **Correcting 矫正错误数据**
  - 发现可能存在错误的数据，对错误数据进行校正，或删除那个样本；检测样本或特征中的异常值，若某个特征没有用处或产生反向的作用，也可以丢弃这个特征
- **Creating 创建新的特征**
  - 将已有特征进行组合、重造等，创建新的特征
- **Charting 可视化**
  - 根据数据的性质和解决方案目标选择正确的可视化图和图表



## 1 问题定义 & 训练集和测试集

通过泰坦尼克号灾难中包含乘客是否生存的训练样本集，设计一个模型用于预测测试数据集中的乘客是否幸存。

**训练集包含信息：**

| PassengerId | Survived | Pclass | Name | Sex  | Age  | SibSp | Parch | Ticket | Fare | Cabin | Embarked |


**测试集包含信息：**

| PassengerId | Pclass | Name | Sex  | Age  | SibSp | Parch | Ticket | Fare | Cabin | Embarked |

| **Variable** | **Definition**                             | **Key**                                        |
| ------------ | ------------------------------------------ | ---------------------------------------------- |
| survival     | Survival                                   | 0 = No, 1 = Yes                                |
| pclass       | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| sex          | Sex                                        |                                                |
| Age          | Age in years                               |                                                |
| sibsp        | # of siblings / spouses aboard the Titanic |                                                |
| parch        | # of parents / children aboard the Titanic |                                                |
| ticket       | Ticket number                              |                                                |
| fare         | Passenger fare                             |                                                |
| cabin        | Cabin number                               |                                                |
| embarked     | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

## 2 整理、准备、清洗数据（特征分析）

### 2.1 导入数据

- 导包

```python
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
```

- 读取数据

```python
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
```

### 2.2 特征分析

- **数据集中所包含的特征**

```python
print(train_df.columns.values)
#---------------------------------------------------------------
'''
['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']
'''
```

- **特征类别**
  - 分类特征
    - 分类：Survived, Sex, and Embarked
    - 序数：Pclass
  - 数值特征
    - 连续：Age, Fare
    - 离散：SibSp, Parch
  - 混合类型
    - Ticket，Cabin 包含字母和数字

```python
train_df.head()
# train_df.tail()
```

| PassengerId | Survived | Pclass | Name | Sex                                               | Age    | SibSp | Parch | Ticket | Fare             | Cabin   | Embarked |      |
| ----------- | -------- | ------ | ---- | ------------------------------------------------- | ------ | ----- | ----- | ------ | ---------------- | ------- | -------- | ---- |
| 0           | 1        | 0      | 3    | Braund, Mr. Owen Harris                           | male   | 22.0  | 1     | 0      | A/5 21171        | 7.2500  | NaN      | S    |
| 1           | 2        | 1      | 1    | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0  | 1     | 0      | PC 17599         | 71.2833 | C85      | C    |
| 2           | 3        | 1      | 3    | Heikkinen, Miss. Laina                            | female | 26.0  | 0     | 0      | STON/O2. 3101282 | 7.9250  | NaN      | S    |

- **可能存在错误的特征**
  - 对大数据集来说，很难查找，一般从表中取出一小部分数据集进行检查
  - 如：姓名中可能包含错误和打字错误

- **存在空值的特征（blank, null or empty）**
  - 如：Cabin, Age, Embarked  存在空值

- **各个特征的数据类型**
  - 训练集：7个特征为正数或浮点数， 测试集：6个
  - 5个特征为字符串类型（object）

```python
train_df.info()
print('_'*40)
test_df.info()

#-----------------------OUTPUT-----------------------------
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
________________________________________
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Name           418 non-null object
Sex            418 non-null object
Age            332 non-null float64
SibSp          418 non-null int64
Parch          418 non-null int64
Ticket         418 non-null object
Fare           417 non-null float64
Cabin          91 non-null object
Embarked       418 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
'''
```

- **数值特征分布**
  - 「帮助在早期确定训练集的代表性」，从下表可知：
  - 总样本数为891，是泰坦尼克号（2,224）上实际乘客人数的40％
  - Survived是0或1的分类特征
  - 大约38％的样本存活，实际存活率为32％
  - 大多数乘客（> 75％）没有与父母或孩子一起旅行
  - 近30％的乘客有兄弟姐妹或配偶
  - 票价差异很大，有很少的 乘客（<1％）支付高达512美元
  - 年龄在65-80岁之间的老年乘客（<1％）很少

```python
train_df.describe()
```

|            | PassengerId | Survived   | Pclass     | Age        | SibSp      | Parch      | Fare       |
| ----------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| count       | 891.000000 | 891.000000 | 891.000000 | 714.000000 | 891.000000 | 891.000000 | 891.000000 |
| mean        | 446.000000 | 0.383838   | 2.308642   | 29.699118  | 0.523008   | 0.381594   | 32.204208  |
| std         | 257.353842 | 0.486592   | 0.836071   | 14.526497  | 1.102743   | 0.806057   | 49.693429  |
| min         | 1.000000   | 0.000000   | 1.000000   | 0.420000   | 0.000000   | 0.000000   | 0.000000   |
| 25%         | 223.500000 | 0.000000   | 2.000000   | 20.125000  | 0.000000   | 0.000000   | 7.910400   |
| 50%         | 446.000000 | 0.000000   | 3.000000   | 28.000000  | 0.000000   | 0.000000   | 14.454200  |
| 75%         | 668.500000 | 1.000000   | 3.000000   | 38.000000  | 1.000000   | 0.000000   | 31.000000  |
| max         | 891.000000 | 1.000000   | 3.000000   | 80.000000  | 8.000000   | 6.000000   | 512.329200 |

- **分类特征分布**
  - 姓名在整个数据集中是唯一的（count = unique = 891）
  - 性别变量为两个值，男性为65％（top=male, freq=577/count=891）
  - Cabin有一些重复的值（几个乘客共用一个小屋）
  - Embarked有三个值。 大多数乘客使用S口（top= S）
  -  船票有很多相同的价格（22%，unique=681）

```python
# 参数include=['O']，表示对Object类型的特征进行统计
train_df.describe(include=['O'])
```

|        | Name   | Sex                        | Ticket | Cabin | Embarked    |
| ------ | -------------------------- | ------ | ----- | ----------- | ---- |
| count  | 891                        | 891    | 891   | 204         | 889  |
| unique | 891                        | 2      | 681   | 147         | 3    |
| top    | Panula, Master. Juha Niilo | male   | 1601  | C23 C25 C27 | S    |
| freq   | 1                          | 577    | 7     | 4           | 644  |

### 2.3 基于特征分析的假设

- **Correlating**
  - 每个特征与生存之间的联系
- **Completing**
  - 填充`Age`特征，因为与是否存活关系紧密
  - 填充`Embarked`特征，因为也与存活或其他重要的特征关系紧密
- **Correcting**
  - `Ticket`特征将会删除，包含高比例的重复项(22%)，并且和存活之间可能没有关联
  - `Cabin`删除，因为含有很多空值
  - `PassengerId`删除，对是否存活的影响很小
  - `Name`可能会删除，因为信息不标准，且可能不会直接有助于生存
- **Creating**
  - 创造新的特征`Family`，基于`Parch`和`SibSp`特征，以得到家庭成员总数
  - 从`Name`特征中提取`Title`作为一个新特征
  - 创建年龄阶层特征`Age bands`，将数字特征转化为分类特征
  - 可能创建票价范围特征`Fare range`，用于帮助分析
- **Classifying**
  - 女性更容易存活
  - 儿童（Age<?）更容易存活
  - 老人更容易存活

### 2.4 表格分析特征

用于快速分析特征相关性（目前只能对没有空值的特征进行此操作）

- **Pclass**: `Pclass=1`和存活(分类#3)之间存在显著相关性(> 0.5)。因此决定在模型中包含这个特性
- **Sex:** 证实了女性的存活率很高，为74%(分类#1)
- **SibSp and Parch**: 这些特征相关性低。最好从这些单独的特征中导出一个或一组特征(创建#1)

```python
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```

|           | Pclass | Survived |
| ------ | -------- | -------- |
| 0      | 1        | 0.629630 |
| 1      | 2        | 0.472826 |
| 2      | 3        | 0.242363 |

```python
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```

|      | Sex    | Survived |
| ---- | ------ | -------- |
| 0    | female | 0.742038 |
| 1    | male   | 0.188908 |

```python
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```

|      | SibSp | Survived |
| ---- | ----- | -------- |
| 1    | 1     | 0.535885 |
| 2    | 2     | 0.464286 |
| 0    | 0     | 0.345395 |
| 3    | 3     | 0.250000 |
| 4    | 4     | 0.166667 |
| 5    | 5     | 0.000000 |
| 6    | 8     | 0.000000 |

```python
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```

|      | Parch | Survived |
| ---- | ----- | -------- |
| 3    | 3     | 0.600000 |
| 1    | 1     | 0.550847 |
| 2    | 2     | 0.500000 |
| 0    | 0     | 0.343658 |
| 5    | 5     | 0.200000 |
| 4    | 4     | 0.000000 |
| 6    | 6     | 0.000000 |

## 3 可视化数据

### 3.1 特征相关性

直方图可用于分析`Age`这样的连续数值变量，可以使用自动定义的区间或等距离范围来指示样本的分布。~~其中条带或范围将有助于识别有用的模式~~。

```python
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
```

<img src="https://raw.githubusercontent.com/shuopic/ImgBed/master/kaggle/1.jpg"/>

- **Observations**
  - 婴儿(年龄≤4岁)存活率高
  - 年龄最大的乘客(年龄= 80岁)幸存下来
  - 大量15-25岁的人没有存活下来
  - 大多数乘客年龄在15-35岁之间
- **Decisions**
  - 在模型训练中，应该考虑年龄特征（假设分类#2）
  - 填充年龄特征中的空值 (completing #1)
  - 为年龄分组 (creating #3)




### 3.2 关联数字和序数特征

使用一个图来组合多个特征来识别相关性

```python
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
```

<img src="https://raw.githubusercontent.com/shuopic/ImgBed/master/kaggle/2.jpg"/>

- **Observations**
  - `Pclass=3`的乘客很多，但是大部分没有幸存下来。确认分类假设#2
  - `Pclass=2`和`Pclass=3`的婴儿乘客大多幸存下来。进一步确定了分类假设#2
  - `Pclass=1`的大多数乘客幸存下来。确认分类假设#3
  - 乘客的年龄分布不同
- **Decisions**
  - 考虑使用`Pclass`特征进行模型训练



### 3.3 关联分类特征

将分类特征与我们的解决方案目标相关联

```python
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
```

<img src="https://raw.githubusercontent.com/shuopic/ImgBed/master/kaggle/3.jpg"/>

- **Observations**
  - 女性乘客的存活率比男性高得多。确认分类(#1)
  - 除`Embarked=C`的男性存活率较高外。这可能是`Pclass`和`Embarked`之间的相关性，反过来是`Pclass`和`Survived`，不一定是`Embarked`和`Survived`之间的直接相关
  - 当C和Q口登船的男性中，`Pclass=3`比`Pclass=2`存活率更高。完成(#2) ？？？
  - 对于`Pclass=3`和男性乘客来说，登机港的存活率各不相同。关联(#1) ？？？



### 3.4 关联分类和数字特征

将分类特征和数值特征相关联。我们可以考虑将`Embarked`（分类特征）、`Sex`（分类特征）、`Fare`（连续数字特征）与`Survived`（分类特征）相关联。

```python
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
```

<img src="https://raw.githubusercontent.com/shuopic/ImgBed/master/kaggle/4.jpg"/>

- **Observations.**
  - 票价越高存活率越高。证实了创建票价范围的假设（#4）
  - `Embarked`与存活率相关。确认#1的关联性和完成#2

- **Decisions.**
  - 考虑对票价分组

## 4 数据整理

目前已经收集了一些解决方案的假设和决策。接下来执行决策和假设，以纠正、创建和完成目标

### 4.1 补全特征值

有以下三种方法用于特征值补全：

1. 一种简单的方法是在均值和标准差之间生成随机数
2. 使用其他特征来猜测缺失值
3. 结合方法1和2。因此，不要根据中位数来猜测年龄值，而是根据不同的类别和性别组合，使用均值和标准差之间的随机数

方法1和3将在我们的模型中引入随机噪声。多次执行的结果可能不同。因此更偏向方法2。

#### 4.1.1 补全Age特征 & 分组

```python
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
```

<img src="https://raw.githubusercontent.com/shuopic/ImgBed/master/kaggle/5.jpg"/>

- 准备一个空数组来包含基于`Pclass x Gender `的组合

```python
guess_ages = np.zeros((2,3))
guess_ages

# ------------------OUTPUT--------------------
array([[0., 0., 0.],
       [0., 0., 0.]])
```

- 根据`Sex`(0 or 1)和`Pclass`(1, 2, 3)两个特征来猜测`Age`
  - 使用搜索结果中符合条件的中位数

```python
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
```

- 对年龄分组

```python
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
```

|      | AgeBand       | Survived |
| ---- | ------------- | -------- |
| 0    | (-0.08, 16.0] | 0.550000 |
| 1    | (16.0, 32.0]  | 0.337374 |
| 2    | (32.0, 48.0]  | 0.412037 |
| 3    | (48.0, 64.0]  | 0.434783 |
| 4    | (64.0, 80.0]  | 0.090909 |

```python
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()
```

|      | Survived | Pclass | Sex  | Age  | SibSp | Parch | Fare    | Embarked | Title | AgeBand      |
| ---- | -------- | ------ | ---- | ---- | ----- | ----- | ------- | -------- | ----- | ------------ |
| 0    | 0        | 3      | 0    | 1    | 1     | 0     | 7.2500  | S        | 1     | (16.0, 32.0] |
| 1    | 1        | 1      | 1    | 2    | 1     | 0     | 71.2833 | C        | 3     | (32.0, 48.0] |
| 2    | 1        | 3      | 1    | 1    | 0     | 0     | 7.9250  | S        | 2     | (16.0, 32.0] |
| 3    | 1        | 1      | 1    | 2    | 1     | 0     | 53.1000 | S        | 3     | (32.0, 48.0] |
| 4    | 0        | 3      | 0    | 2    | 0     | 0     | 8.0500  | S        | 1     | (32.0, 48.0] |

- 删除`AgeBand`特征

```python
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
```

#### 4.1.2 补全Embarked特征

在`Embarked`特征中丢失了两个值，将样本中出现最多的值填进去

- 统计出现最多的值

```python
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

# ---------------OUTPUT------------------
'S'
```

- 补全

```python
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```

|      | Embarked | Survived |
| ---- | -------- | -------- |
| 0    | C        | 0.553571 |
| 1    | Q        | 0.389610 |
| 2    | S        | 0.339009 |

#### 4.1.3 补全Fare特征

使用中位数进行补全

```python
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()
```

|      | PassengerId | Pclass | Sex  | Age  | Fare    | Embarked | Title | IsAlone | Age*Class |
| ---- | ----------- | ------ | ---- | ---- | ------- | -------- | ----- | ------- | --------- |
| 0    | 892         | 3      | 0    | 2    | 7.8292  | 2        | 1     | 1       | 6         |
| 1    | 893         | 3      | 1    | 2    | 7.0000  | 0        | 3     | 0       | 6         |
| 2    | 894         | 2      | 0    | 3    | 9.6875  | 2        | 1     | 1       | 6         |
| 3    | 895         | 3      | 0    | 1    | 8.6625  | 0        | 1     | 1       | 3         |
| 4    | 896         | 3      | 1    | 1    | 12.2875 | 0        | 3     | 0       | 3         |

### 4.2 创建新特征

#### 4.2.1 创建title特征

先分析是否可以将`Name`特征来提取`titles`并测试`titles`和`Survival`之间的相关性，然后再删除`Name`和`PassengerId`特征。

提取`Name`中的第一个词：

```python
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
```

| Sex      | female | male |
| -------- | ------ | ---- |
| Title    |        |      |
| Capt     | 0      | 1    |
| Col      | 0      | 2    |
| Countess | 1      | 0    |
| Don      | 0      | 1    |
| Dr       | 1      | 6    |
| Jonkheer | 0      | 1    |
| Lady     | 1      | 0    |
| Major    | 0      | 2    |
| Master   | 0      | 40   |
| Miss     | 182    | 0    |
| Mlle     | 2      | 0    |
| Mme      | 1      | 0    |
| Mr       | 0      | 517  |
| Mrs      | 125    | 0    |
| Ms       | 1      | 0    |
| Rev      | 0      | 6    |
| Sir      | 0      | 1    |

- **Observations.**
  - 大多数`title`与年龄分组想对应
  - 一些`title`大多数存活下来（Mme, Lady, Sir），一些没存活（Don, Rev, Jokheer）

- **Decision.**
  - 决定保留新特征`title`用于模型训练

将`title`特征进行合并：

```python
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
```

|      | Title  | Survived |
| ---- | ------ | -------- |
| 0    | Master | 0.575000 |
| 1    | Miss   | 0.702703 |
| 2    | Mr     | 0.156673 |
| 3    | Mrs    | 0.793651 |
| 4    | Rare   | 0.347826 |

将`title`分类特征转换为序数特征：

```python
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
```

|      | PassengerId | Survived | Pclass | Name                                              | Sex    | Age  | SibSp | Parch | Fare    | Embarked | Title |
| ---- | ----------- | -------- | ------ | ------------------------------------------------- | ------ | ---- | ----- | ----- | ------- | -------- | ----- |
| 0    | 1           | 0        | 3      | Braund, Mr. Owen Harris                           | male   | 22.0 | 1     | 0     | 7.2500  | S        | 1     |
| 1    | 2           | 1        | 1      | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0 | 1     | 0     | 71.2833 | C        | 3     |
| 2    | 3           | 1        | 3      | Heikkinen, Miss. Laina                            | female | 26.0 | 0     | 0     | 7.9250  | S        | 2     |
| 3    | 4           | 1        | 1      | Futrelle, Mrs. Jacques Heath (Lily May Peel)      | female | 35.0 | 1     | 0     | 53.1000 | S        | 3     |
| 4    | 5           | 0        | 3      | Allen, Mr. William Henry                          | male   | 35.0 | 0     | 0     | 8.0500  | S        | 1     |



现在可以删除`Name`和`PassengerId`特征了：

```python
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

# -------------OUTPUT--------------
'''
((891, 9), (418, 9))
'''
```

#### 4.2.2 创建FamilySize特征

将`Parch`（父母+孩子），`SibSp`（兄妹+配偶）

```python
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
```

|             | FamilySize | Survived |
| ---------- | -------- | -------- |
| 3          | 4        | 0.724138 |
| 2          | 3        | 0.578431 |
| 1          | 2        | 0.552795 |
| 6          | 7        | 0.333333 |
| 0          | 1        | 0.303538 |
| 4          | 5        | 0.200000 |
| 5          | 6        | 0.136364 |
| 7          | 8        | 0.000000 |
| 8          | 11       | 0.000000 |

#### 4.2.3 创建IsAlone特征

```python
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
```

|         | IsAlone | Survived |
| ------- | -------- | -------- |
| 0       | 0        | 0.505650 |
| 1       | 1        | 0.303538 |

#### 4.2.4 创建组合特征Age\*Class

将`Pclass`和`Age`特征进行组合

```python
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
```

#### 4.2.5 创建FareBand特征

- 对票价进行分组
  - 分组依据：每组中含有相同数量的样本

```python
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
```

|      | FareBand        | Survived |
| ---- | --------------- | -------- |
| 0    | (-0.001, 7.91]  | 0.197309 |
| 1    | (7.91, 14.454]  | 0.303571 |
| 2    | (14.454, 31.0]  | 0.454955 |
| 3    | (31.0, 512.329] | 0.581081 |

- 将分组范围转换为数字类型

```python
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)
```

|     | Survived | Pclass | Sex  | Age  | Fare | Embarked | Title | IsAlone | Age*Class |
| ---- | ---- | ---- | ---- | ---- | ----- | ----- | ---- | ---- | ---- |
| 0        | 0      | 3    | 0    | 1    | 0        | 0     | 1       | 0         | 3    |
| 1        | 1      | 1    | 1    | 2    | 3        | 1     | 3       | 0         | 2    |
| 2        | 1      | 3    | 1    | 1    | 1        | 0     | 2       | 1         | 3    |
| 3        | 1      | 1    | 1    | 2    | 3        | 0     | 3       | 0         | 2    |
| 4        | 0      | 3    | 0    | 2    | 1        | 0     | 1       | 1         | 6    |
| 5        | 0      | 3    | 0    | 1    | 1        | 2     | 1       | 1         | 3    |
| 6        | 0      | 1    | 0    | 3    | 3        | 0     | 1       | 1         | 3    |
| 7        | 0      | 3    | 0    | 0    | 2        | 0     | 4       | 0         | 0    |
| 8        | 1      | 3    | 1    | 1    | 1        | 0     | 3       | 0         | 3    |
| 9        | 1      | 2    | 1    | 0    | 2        | 1     | 3       | 0         | 0    |

### 4.3 特征转换

将所有非数字特征转换为数字特征。这是大多数模型算法所要求的，这样做也将有助于完成目标。

#### 4.3.1 转换Sex特征

将`Sex`特征转换为`Gender`新特征，其中`female=1, male=0`

```python
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head(
```

#### 4.3.2 转换Embarked特征

将`Embarked`中的`S, C, Q`特征值转换为数字特征

```python
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
```

|     | Survived | Pclass | Sex  | Age  | Fare | Embarked | Title | IsAlone | Age*Class |
| -------- | ------ | ---- | ---- | ---- | -------- | ----- | ------- | --------- | ---- |
| 0        | 0      | 3    | 0    | 1    | 7.2500   | 0     | 1       | 0         | 3    |
| 1        | 1      | 1    | 1    | 2    | 71.2833  | 1     | 3       | 0         | 2    |
| 2        | 1      | 3    | 1    | 1    | 7.9250   | 0     | 2       | 1         | 3    |
| 3        | 1      | 1    | 1    | 2    | 53.1000  | 0     | 3       | 0         | 2    |
| 4        | 0      | 3    | 0    | 2    | 8.0500   | 0     | 1       | 1         | 6    |

### 4.4 删除特征

删除没用（或用处不大）的特征以减少数据量，加快分析速度

#### 4.4.1 删除Cabin和Ticket特征

- 根据前面的分析，决定删除`Cabin`和`Ticket`特征

```python
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

# ----------------------OUTPUT--------------------------
'''
Before (891, 12) (418, 11) (891, 12) (418, 11)
('After', (891, 10), (418, 9), (891, 10), (418, 9))
'''
```

#### 4.4.2 删除Parch、 SibSp和FamilySize特征

由于4.1.3节中创建了`IsAlone`特征，比这三个特征更具有代表意义，因此把这三个特征删除。（FamilySize是否可以保留？）

```python
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()
```

## 5 模型预测 & 分析 & 检验

### 5.1 模型预测

- 现在，准备训练一个模型并预测所需的解决方案。有60多种预测建模算法可供选择。
- 必须了解问题的类型和解决方案需求，以缩小到我们可以评估的几个模型
  - 我们的问题是分类和回归问题，想确定输出(存活与否)与其他特征之间的关联
  - 当我们用给定的数据集训练我们的模型时，我们也在进行一种叫做监督学习的机器学习
  - 有了这两个标准——监督学习加上分类和回归，我们可以将模型的选择范围缩小到几个
  - 其中包括：
    - Logistic Regression
    - KNN or k-Nearest Neighbors
    - Support Vector Machines
    - Naive Bayes classifier
    - Decision Tree
    - Random Forrest
    - Perceptron
    - Artificial neural network
    - RVM or Relevance Vector Machine

```python
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

#-------------------OUTPUT--------------------
'''
((891, 8), (891,), (418, 8))
'''
```

#### 5.1.1 Logistic Regression

逻辑回归是工作流早期运行的有用模型。逻辑回归通过使用逻辑函数(即累积逻辑分布)估计概率来测量分类因变量(特征)和一个或多个自变量(特征)之间的关系。

```python
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
```

#### 5.1.2 KNN or k-Nearest Neighbors

在模式识别中，k-最近邻算法(简称k-NN)是一种用于分类和回归的非参数方法。样本由其邻居的多数票进行分类，样本被分配给其k个最近邻居中最常见的类别(k是正整数，通常很小)。如果k = 1，那么该对象被简单地分配给该单个最近邻居的类。

KNN的置信分数比逻辑回归要好，但比SVM差。

```python
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
```

#### 5.1.3 Support Vector Machines

使用支持向量机建模，支持向量机是监督学习模型，具有分析用于分类和回归分析的数据的相关学习算法。给定一组训练样本，每个样本标记为属于两个类别中的一个或另一个，SVM训练算法建立一个模型，将新的测试样本分配给一个类别或另一个类别，使其成为非概率二进制线性分类器。

```python
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
```

#### 5.1.4 Naive Bayes classifier

在机器学习中，朴素贝叶斯分类器是一个简单的概率分类器家族，基于贝叶斯定理，特征之间具有强(朴素)独立性假设。朴素贝叶斯分类器具有很高的可伸缩性，在学习问题中需要多个变量(特征)呈线性的参数。参考维基百科。

```python
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
```

#### 5.1.5 Decision Tree

该模型使用决策树作为预测模型，将特征(树枝)映射到关于目标值(树叶)的结论。目标变量可以取一组有限值的树模型称为分类树；在这些树结构中，树叶代表类标签，树枝代表导致这些类标签的特征的结合。目标变量可以取连续值(通常是实数)的决策树称为回归树。

```python
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
```

#### 5.1.6 Random Forrest

下一个模型随机森林是最受欢迎的模型之一。随机森林或随机决策森林是一种用于分类、回归和其他任务的集成学习方法，它通过在训练时构建大量决策树(`n _ assessors = 100`)并输出作为单个树的类别(分类)或平均预测(回归)模式的类别来操作。

```python
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
```

#### 5.1.7 Perceptron

感知器是一种用于二进制分类器监督学习的算法(可以决定由数字向量表示的输入是否属于某个特定类别的函数)。它是一种线性分类器，即基于将一组权重与特征向量相结合的线性预测函数进行预测的分类算法。该算法允许在线学习，因为它一次处理一个训练集中的元素。

```python
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
```

#### 5.1.8 Linear SVC

```python
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
```

#### 5.1.9 Stochastic Gradient Descent

```python
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
```

### 5.2 分析

可以使用Logistic回归来验证我们对功能创建和完成目标的假设和决策。 这可以通过计算决策函数中的特征的系数来完成。

正系数增加了响应的对数几率（从而增加了概率），负系数降低了响应的对数几率（从而降低了概率）。

```python
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
```

|      | Feature   | Correlation |
| ---- | --------- | ----------- |
| 1    | Sex       | 2.201527    |
| 5    | Title     | 0.398234    |
| 2    | Age       | 0.287163    |
| 4    | Embarked  | 0.261762    |
| 6    | IsAlone   | 0.129140    |
| 3    | Fare      | -0.085150   |
| 7    | Age*Class | -0.311200   |
| 0    | Pclass    | -0.749007   |

### 5.3 模型评估

现在可以对所有模型的评估进行排序，以选择最适合我们问题的模型。虽然决策树和随机森林得分相同，但我们选择使用随机森林，因为它们纠正了决策树过度拟合其训练集的习惯。

```python
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
```

|          | Model | Score                      |
| ----- | -------------------------- | ----- |
| 3     | Random Forest              | 86.76 |
| 8     | Decision Tree              | 86.76 |
| 1     | KNN                        | 84.74 |
| 0     | Support Vector Machines    | 83.84 |
| 2     | Logistic Regression        | 80.36 |
| 7     | Linear SVC                 | 79.12 |
| 6     | Stochastic Gradient Decent | 78.56 |
| 5     | Perceptron                 | 78.00 |
| 4     | Naive Bayes                | 72.28 |

## 6 提交结果

```python
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)
```

