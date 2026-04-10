import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer,KNNImputer     
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostRegressor,AdaBoostClassifier   
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer


# 1.import data
train_path1=r'.\XXX.csv' #RandomOverSampler generated dataset
train_path2=r'.\XXX.csv' #SMOTE generated dataset
train_path3=r'.\XXX.csv' #ADASYN generated dataset
test_path=r'.\XXX.csv'  #test set

df1=pd.read_csv(train_path1,header=0,encoding="gbk")
x_train1=df1.iloc[:,[1,2,3,4,5,6,7]].values  
y_train1=df1.iloc[:,-1].values  

df2=pd.read_csv(train_path2,header=0,encoding="gbk")
x_train2=df2.iloc[:,[1,2,3,4,5,6,7]].values  
y_train2=df2.iloc[:,-1].values  

df3=pd.read_csv(train_path3,header=0,encoding="gbk")
x_train3=df3.iloc[:,[1,2,3,4,5,6,7]].values 
y_train3=df3.iloc[:,-1].values  

df=pd.read_csv(test_path,header=0,encoding="gbk")
x_test=df.iloc[:,[1,3,4,5,6,7,8]].values 
y_test=df.iloc[:,-1].values

X=[x_train1,x_train2,x_train3]
y=[y_train1,y_train2,y_train3]

# 2.data augmentation method compare
f1v = [] #cross validation results
f1t = []  #test set results
for i in range(len(X)):
    
    #estimator = RandomForestClassifier(random_state=0,n_estimators=50,criterion='entropy',max_depth=7, max_features=3)
    estimator = RandomForestClassifier(n_estimators=120)
    #scores = cross_val_score(estimator,X[i],y[i],scoring='f1_macro',cv=10).mean()
    
    strKFold = StratifiedKFold(n_splits=3,shuffle=True,random_state=0)
    scores = cross_val_score(estimator,X[i],y[i],scoring='f1_macro',cv=strKFold).mean()
    f1v.append(scores)
    estimator.fit(X[i],y[i])
    yprd_test=estimator.predict(x_test)
    
    #average:'macro','weighted','micro','samples'
    pre1, rec1, f11, _ = precision_recall_fscore_support(y_test, yprd_test,average='macro')
    print('test set: precision {:.4f}  recall {:.4f} F1 {:.4f}'.format(np.mean(pre1),np.mean(rec1),np.mean(f11)))
    f1t.append(f11)
    
    sns.set()
    f3,ax3=plt.subplots()
    cm_test2= confusion_matrix(y_test, yprd_test, labels=[0,1,2])
    acc3=(np.trace(cm_test2))/np.sum(np.sum(cm_test2))
    print(acc3)
    tt3='accuracy'+str(round(acc3*100,2))+'%'
    sns.heatmap(cm_test2,annot=True,ax=ax3) 
    ax3.set_title(tt3) 
    ax3.set_xlabel('predicted') 
    ax3.set_ylabel('true')   