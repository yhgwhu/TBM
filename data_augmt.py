from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostRegressor,AdaBoostClassifier  
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler,SMOTENC,BorderlineSMOTE 
from collections import Counter
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def SMOTENC_n(X,y,lst,n):
    smote_nc = SMOTENC(categorical_features=lst, random_state=0)  
    xSMOTENC, ySMOTENC = smote_nc.fit_resample(X,y)
    for i in range(n-1):
        xSMOTENC=np.concatenate([xSMOTENC,X],axis=0)
        ySMOTENC=np.concatenate([ySMOTENC,y],axis=0)
        
        xSMOTENC, ySMOTENC = smote_nc.fit_resample(xSMOTENC,ySMOTENC)      
    
    return xSMOTENC, ySMOTENC

def ROS_n(X,y,n):
    xROS,yROS = RandomOverSampler().fit_resample(X,y)
    for i in range(n-1):
        xROS=np.concatenate([xROS,X],axis=0)
        yROS=np.concatenate([yROS,y],axis=0)
        
        xROS, yROS = RandomOverSampler().fit_resample(xROS,yROS)
          
    return xROS, yROS
    
    
# 1.import data
train_path=r'.\XXX.csv' #filled dataset by multiple imputation
test_path=r'.\XXX.csv'  #test set
df1=pd.read_csv(train_path,header=0,encoding="gbk")
x_train=df1.iloc[:,[1,2,3,4,5,6,7,8]].values  
y_train=df1.iloc[:,-1].values  

df2=pd.read_csv(test_path,header=0,encoding="gbk")
x_test=df2.iloc[:,[1,2,3,4,5,6,7,8]].values 
y_test=df2.iloc[:,-1].values

result=sorted(Counter(y_train).items())
print("original dataset sample:")
for i in result:
    print(f"class{i[0]} has {i[1]} samples")

# 2.data augmentation   
#RandomOverSampler
xROS,yROS=RandomOverSampler().fit_resample(x_train,y_train)
result0=sorted(Counter(yROS).items()) 
print("RandomOverSampler processed：")
for i in result0:
    print(f"class {i[0]} has {i[1]} samples")

#SMOTENC
smote_nc = SMOTENC(categorical_features=[1], random_state=0)  
xSMOTENC, ySMOTENC = smote_nc.fit_resample(x_train,y_train)
result1=sorted(Counter(ySMOTENC).items())
print("SMOTENC processed：")
for i in result1:
    print(f"class {i[0]}has {i[1]} samples")

xSMOTENC2, ySMOTENC2=SMOTENC_n(x_train,y_train,[1],2)
result2=sorted(Counter(ySMOTENC2).items())
print("SMOTENC2 processed：")
for i in result2:
    print(f"class {i[0]} has {i[1]} samples")
    
xROS2, yROS2=ROS_n(x_train,y_train,2)
result3=sorted(Counter(yROS2).items())
print("RandomOverSample2 processed：")
for i in result3:
    print(f"class {i[0]} has {i[1]} samples")
    
xSMOTENC3, ySMOTENC3=SMOTENC_n(x_train,y_train,[1],3)
xROS3, yROS3=ROS_n(x_train,y_train,3)

xSMOTENC4, ySMOTENC4=SMOTENC_n(x_train,y_train,[1],4)
xROS4, yROS4=ROS_n(x_train,y_train,4)

xSMOTENC5, ySMOTENC5=SMOTENC_n(x_train,y_train,[1],5)
xROS5, yROS5=ROS_n(x_train,y_train,5)
    
X=[x_train,xROS,xSMOTENC,xROS2,xSMOTENC2,xROS3,xSMOTENC3,xROS4,xSMOTENC4,xROS5,xSMOTENC5]
y=[y_train,yROS,ySMOTENC,yROS2,ySMOTENC2,yROS3,ySMOTENC3,yROS4,ySMOTENC4,yROS5,ySMOTENC5]
    
f1_v=[]  #cross validation results
f1_t=[]  #test set results

for i in range(len(X)):
    
    #cross validation
    estimator = RandomForestClassifier(random_state=0,n_estimators=250,criterion='entropy')
    scores = cross_val_score(estimator,X[i],y[i],scoring='f1_macro',cv=5).mean()
    f1_v.append(scores)
    
    #test set
    estimator.fit(X[i],y[i])
    yprd_test=estimator.predict(x_test)

    #average:'macro','weighted','micro','samples'
    pre1, rec1, f11, _ = precision_recall_fscore_support(y_test, yprd_test,average='macro')
    print('test set: precision {:.4f}  recall {:.4f} F1 {:.4f}'.format(np.mean(pre1),np.mean(rec1),np.mean(f11)))
    f1_t.append(f11)
    
    #confusion matrix
    sns.set()
    f3,ax3=plt.subplots(dpi=500)
    cm_test2= confusion_matrix(y_test, yprd_test, labels=[0,1,2])
    acc3=(np.trace(cm_test2))/np.sum(np.sum(cm_test2))
    print(acc3)
    tt3='accuracy'+str(round(acc3*100,2))+'%'
    sns.heatmap(cm_test2,annot=True,ax=ax3,cmap=plt.cm.Blues) 
    #ax3.set_title(tt3) 
    ax3.set_xlabel('predicted') 
    ax3.set_ylabel('true')  

# 3. data augmentation method compare    
[*zip(["original","RandomOverSampler","SMOTENC","RandomOverSampler-2","SMOTENC-2","RandomOverSampler-3","SMOTENC-3","RandomOverSampler-4","SMOTENC-4","RandomOverSampler-5","SMOTENC-5"],f1_v)]
[*zip(["original","RandomOverSampler","SMOTENC","RandomOverSampler-2","SMOTENC-2","RandomOverSampler-3","SMOTENC-3","RandomOverSampler-4","SMOTENC-4","RandomOverSampler-5","SMOTENC-5"],f1_t)]

x_labels = ["original","RandomOverSampler","SMOTENC","RandomOverSampler-2","SMOTENC-2","RandomOverSampler-3","SMOTENC-3","RandomOverSampler-4","SMOTENC-4","RandomOverSampler-5","SMOTENC-5"]
colors = ['r','g','b','y','m','r','g','b','y','m','navy']


plt.figure(figsize=(12,6)) 
ax = plt.subplot(111)  

for i in np.arange(len(f1_v)):
    ax.barh(i,f1_v[i],color=colors[i],alpha=0.6,align='center')   
ax.set_title('Data augmentation: Validation set')
ax.set_xlim(left=np.min(f1_v)*0.9,right=np.max(f1_v)*1.1) 
ax.set_yticks(np.arange(len(f1_v)))  
ax.set_xlabel('f1')
ax.set_yticklabels(x_labels) 
plt.show()

plt.figure(figsize=(12,6)) 
ax = plt.subplot(111)  
for i in np.arange(len(f1_t)):
    ax.barh(i,f1_t[i],color=colors[i],alpha=0.6,align='center')  
ax.set_title('Data augmentation: Test set')
ax.set_xlim(left=np.min(f1_t)*0.9,right=np.max(f1_t)*1.1)  
ax.set_yticks(np.arange(len(f1_t)))  
ax.set_xlabel('f1')     
ax.set_yticklabels(x_labels) 
plt.show()

