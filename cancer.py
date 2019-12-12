#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'joe'

#导入pandas跟numpy工具包

import pandas as pd
import numpy as np
"""
#创建特征列表
column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion',
                'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

#使用pandas.read_csv 函数从互联网读取指定数据
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                   names=column_names)

#将？替换为标准缺失值表示。
data = data.replace(to_replace='?',value=np.nan)
#丢弃带有缺失值的数据（只要有一个维度有缺失）
data = data.dropna(how='any')
#输出data的数据量和维度
#data.shape

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr = LogisticRegression()
sgdc = SGDClassifier()

lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)

sgdc.fit(X_train,y_train)
sgdc_y_predict = sgdc.predict(X_test)

from sklearn.metrics import classification_report

a = lr.score(X_test,y_test)
print  'Accuracy of LR Classifier:',lr.score(X_test,y_test)

b = classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant'])
print classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant'])

c = sgdc.score(X_test,y_test)
print 'Accuarcy of SGD Classifier:',sgdc.score(X_test,y_test)

d = classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant'])
print classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant'])
"""
"""
#手写体数据分割代码样例
from sklearn.datasets import load_digits

digits = load_digits()

digits.data.shape

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)

y_train.shape

y_train.shape

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC()

lsvc.fit(X_train,y_train)
y_predict = lsvc.predict(X_test)

a = lsvc.score(X_test,y_test)
print 'The Accuracy of Linear SVC is:',a

from sklearn.metrics import classification_report

b = classification_report(y_test,y_predict,target_names=digits.target_names.astype(str))

print b
"""

#20类新闻文本的数据细节

from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')

print len(news.data)
print news.data[0]


print ccs




