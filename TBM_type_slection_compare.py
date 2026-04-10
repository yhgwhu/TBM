from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostRegressor,AdaBoostClassifier   
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler,SMOTENC,BorderlineSMOTE 
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
    
# 1. import data
train_path=r'.\XXX.csv' #augmented dataset
test_path=r'.\XXX.csv' #test set
test_path1=r'.\XXX.csv' #type selection failure cases

df1=pd.read_csv(train_path,header=None,encoding="utf-8")
x_train=df1.iloc[:,0:-1].values  
y_train=df1.iloc[:,-1].values  

df2=pd.read_csv(test_path,header=0,encoding="utf-8")
x_test=df2.iloc[:,[1,2,3,4,5,6,7,8]].values 
y_test=df2.iloc[:,-1].values

df3=pd.read_csv(test_path1,header=0,encoding="utf-8")
x_test1=df3.iloc[:,[1,2,3,4,5,6,7,8]].values 
#y_test=df2.iloc[:,-1].values
  
# 2.TBM type selection model compare
f1_v=[]  #cross validation results
f1_t=[]  #test set results
#cross validation
scoring = ['precision_macro', 'recall_macro','f1_macro'] 
estimator = XGBClassifier(n_estimators=250, max_depth=5)
#estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=250)
#estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=250,learning_rate=1.5,algorithm="SAMME")
#estimator = KNeighborsClassifier(n_neighbors=8, weights='distance')  #uniform
#estimator = svm.SVC(C=5, kernel='linear')
#estimator = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, max_features=8)    
#estimator = RandomForestClassifier(random_state=0,n_estimators=250,criterion='entropy')
#estimator = lgb.LGBMClassifier(n_estimators=50,boosting_type='gbdt',learning_rate=0.1)
scores = cross_validate(estimator,x_train,y_train,scoring=scoring,cv=5)
p0=scores['test_precision_macro']
p0=np.array(p0)
r0=scores['test_recall_macro']
r0=np.array(r0)
f10=scores['test_f1_macro']
f10=np.array(f10)

res0=[p0.mean(),r0.mean(),f10.mean()]
#print(scores['test_precision_macro'])
f1_v.append(res0)
f1_v=np.array(f1_v)

#test set
estimator.fit(x_train,y_train)
yprd_test=estimator.predict(x_test)
yprd_test1=estimator.predict(x_test1)
print(yprd_test1)

# 3.model evaluation
#average:'macro','weighted','micro','samples'
pre, rec, f1, _ = precision_recall_fscore_support(y_test, yprd_test,average='macro')
print('test set: precision {:.4f}  recall {:.4f} F1 {:.4f}'.format(np.mean(pre),np.mean(rec),np.mean(f1)))
res=[pre,rec,f1]
f1_t.append(res)
f1_t=np.array(f1_t)

sns.set()
f3,ax3=plt.subplots(dpi=200)
cm_test2= confusion_matrix(y_test, yprd_test, labels=[0,1,2])
acc3=(np.trace(cm_test2))/np.sum(np.sum(cm_test2))
print(acc3)
tt3='accuracy'+str(round(acc3*100,2))+'%'
sns.heatmap(cm_test2,annot=True,ax=ax3,cmap=plt.cm.Blues) 
ax3.set_xlabel('predicted') 
ax3.set_ylabel('true') 