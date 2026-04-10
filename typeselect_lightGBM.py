import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1.import data
train_path=r'.\XXX.csv' #augmented dataset
test_path=r'.\XXX.csv'  #test set

df0=pd.read_csv(train_path,header=0,encoding="gbk")
x_train=df0.iloc[:,[1,2,3,4,5,6,7,8]].values  
y_train=df0.iloc[:,-1].values  

df=pd.read_csv(test_path,header=0,encoding="gbk")
x_test=df.iloc[:,[1,2,3,4,5,6,7,8]].values 
y_test=df.iloc[:,-1].values

# 2.train model
X_t, X_v, y_t, y_v = train_test_split(x_train, y_train, test_size = 0.30, shuffle=True,random_state = 666)
classifier = lgb.LGBMClassifier(n_estimators=100,boosting_type='gbdt',learning_rate=0.25,importance_type='gain')
classifier.fit(X_t, y_t)
feature_importance=classifier.feature_importances_  
y_pred = classifier.predict(X_v)
y_testprba = classifier.predict_proba(X_v)[:,1] 
y_trainpred = classifier.predict(X_t)
y_trainprba = classifier.predict_proba(X_t)[:,1]

# 3.evaluate model
cm_test = confusion_matrix(y_v, y_pred,labels=[0,1,2])
cm_train = confusion_matrix(y_t, y_trainpred,labels=[0,1,2])
sns.set()
f1,ax1=plt.subplots()
acc1=(np.trace(cm_train))/np.sum(np.sum(cm_train))
print(acc1)
tt1='accuracy'+str(round(acc1*100,2))+'%'
sns.heatmap(cm_train,annot=True,ax=ax1)
ax1.set_title(tt1) 
ax1.set_xlabel('predicted') 
ax1.set_ylabel('true') 

sns.set()
f2,ax2=plt.subplots()
acc2=(np.trace(cm_test))/np.sum(np.sum(cm_test))
print(acc2)
tt2='accuracy'+str(round(acc2*100,2))+'%'
sns.heatmap(cm_test,annot=True,ax=ax2) 
ax2.set_title(tt2)
ax2.set_xlabel('predicted') 
ax2.set_ylabel('true')

y_testpred = classifier.predict(x_test)
y_testprba1 = classifier.predict_proba(x_test)[:,1] 
cm_test1 = confusion_matrix(y_test, y_testpred,labels=[0,1,2])
sns.set()
f3,ax3=plt.subplots()
acc3=(np.trace(cm_test1))/np.sum(np.sum(cm_test1))
print(acc3)
tt3='accuracy'+str(round(acc3*100,2))+'%'
sns.heatmap(cm_test1,annot=True,ax=ax3) 
ax3.set_title(tt3) 
ax3.set_xlabel('predicted') 
ax3.set_ylabel('true') 
