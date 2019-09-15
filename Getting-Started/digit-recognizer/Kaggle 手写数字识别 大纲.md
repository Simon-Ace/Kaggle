- 载入数据
  - 使用pandas read csv
  - 使用seaborn查看下数据分布
  - pandas查看是否存在空数据
  - 正则化，更改数据范围
  - 更改数据维度
  - keras，将Y标签变成one hot
  - Keras，将一部分训练数据变成val验证集
- 构建网络
  - 网络结构
    - Conv2D 5x5x32
    - Conv2D 5x5x32
    - maxpool,  size 2,2
    - dropout 0.25
    - Conv2D 3x3x64
    - Conv2D 3x3x64
    - maxpool,  size 2,2  strides 2,2
    - dropout 0.25
    - Flatten
    - full connect 256
    - dropout 0.5
    - full connect 10
  - 优化项
    - optimizer：RMSprop
    - compile the model
    - learning rate annealer
    - epchs 30  batch_size 86
    - ImageDataGenerator
    - fit generator （训练）
  - 评估模型
    - Training loss、validation loss、Training accuracy、Validation accuracy



---

有空值得时候怎么办？

X_train.isnull().any().describe()

```
# 添加了两个空值
count       784
unique        2
top       False
freq        782
dtype: object
```

```
# 没有空值
count       784
unique        1
top       False
freq        784
dtype: object
```





Z:\workdir\SpackMLlib