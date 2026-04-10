import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer,KNNImputer     
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostRegressor,AdaBoostClassifier   
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
import seaborn as sns
from sklearn.impute import IterativeImputer
from sklearn.svm import SVR   
from sklearn.utils import shuffle

# 1.import data
train_path0=r'.\XXX.csv' #cases with complete data
train_path=r'.\XXX.csv' # all cases (complete and incomplete)
test_path=r'.\XXX.csv' #test set (complete cases)

df0=pd.read_csv(train_path0,header=0,encoding="gbk")
df0 = shuffle(df0,random_state=0)
x_train0=df0.iloc[:,1:-1].values  #features
y_train0=df0.iloc[:,-1].values  #label

df1=pd.read_csv(train_path,header=0,encoding="gbk")
df1 = shuffle(df1,random_state=0)
x_train=df1.iloc[:,1:-1].values  
y_train=df1.iloc[:,-1].values  

df2=pd.read_csv(test_path,header=0,encoding="gbk")
df2 = shuffle(df2,random_state=0)
x_test=df2.iloc[:,1:-1].values 
y_test=df2.iloc[:,-1].values

# 2.compare imputation methods
# mean impute
imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')  
X_missing_mean = imp_mean.fit_transform(x_train)
pd.DataFrame(X_missing_mean).isnull().sum()

# zero impute
imp_0 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0) 
X_missing_0 = imp_0.fit_transform(x_train)

# median impute
imp_md = SimpleImputer(missing_values=np.nan,strategy='median') 
X_missing_md = imp_md.fit_transform(x_train)

#hot deck imputation
imp_hd = KNNImputer(n_neighbors=1)
X_missing_hd=imp_hd.fit_transform(x_train)

#multiple imputation
#estimator = BayesianRidge() 
#estimator = DecisionTreeRegressor(max_features='sqrt', random_state=0) #max_features='sqrt'
#estimator = RandomForestRegressor(n_estimators=50)
estimator = SVR()
#estimator = KNeighborsRegressor(n_neighbors=5)
#estimator = AdaBoostRegressor(n_estimators=50)
#estimator = XGBRegressor(n_estimators=50)
Iter= IterativeImputer(estimator=estimator,max_iter=50,random_state=2024)
X_missing_itr=Iter.fit_transform(x_train)    
imputer = KNNImputer(n_neighbors=5)
X_missing_knn=imputer.fit_transform(x_train)

#X = [X_missing_mean,X_missing_0,X_missing_md,X_missing_itr,X_missing_reg,X_missing_knn]
#X = [X_missing_mean,X_missing_0,X_missing_md,X_missing_itr,X_missing_reg]
X = [X_missing_mean,X_missing_0,X_missing_md,X_missing_hd,X_missing_itr,X_missing_knn]
f1v = [] #cross validation results
f1t = []  #test set results

# no missing data
#estimator0 = RandomForestClassifier(random_state=0,n_estimators=200,criterion='entropy',max_depth=5 ,max_features=7)
estimator0 = RandomForestClassifier(random_state=0,n_estimators=50,criterion='entropy')
scores = cross_val_score(estimator0,x_train0,y_train0,scoring='f1_macro',cv=5).mean()
f1v.append(scores)

estimator0.fit(x_train0,y_train0)
yprd_test0=estimator0.predict(x_test)
pre10, rec10, f110, _ = precision_recall_fscore_support(y_test, yprd_test0,average='macro')
f1t.append(f110)

for i in X:
    estimator = RandomForestClassifier(random_state=0,n_estimators=250,criterion='entropy')
    #estimator = AdaBoostClassifier(n_estimators=50)
    scores = cross_val_score(estimator,i,y_train,scoring='f1_macro',cv=5).mean()
    f1v.append(scores)
    estimator.fit(i,y_train)
    yprd_test=estimator.predict(x_test)

    #average:'macro','weighted','micro','samples'
    pre1, rec1, f11, _ = precision_recall_fscore_support(y_test, yprd_test,average='macro')
    print('test set: precision {:.4f}  recall {:.4f} F1 {:.4f}'.format(np.mean(pre1),np.mean(rec1),np.mean(f11)))
    f1t.append(f11)
    
    #plot confusion matrix
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

# 3.evaluate model's performance (i.e., imputation method's performance)
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

#[*zip(["X_full","X_missing_mean","X_missing_0","X_missing_md","X_missing_itr","X_missing_reg","X_missing_knn"],f1v)]
#[*zip(["X_full","X_missing_mean","X_missing_0","X_missing_md","X_missing_itr","X_missing_reg","X_missing_knn"],f1t)]

[*zip(["X_full","X_missing_mean","X_missing_0","X_missing_md","X_missing_hd","X_missing_itr","X_missing_knn","X_missing_itr1"],f1v)]
[*zip(["X_full","X_missing_mean","X_missing_0","X_missing_md","X_missing_hd","X_missing_itr","X_missing_knn","X_missing_itr1"],f1t)]

#x_labels = ['Full data','Mean Imputation','Zero Imputation','Median Imputation','Iterative Imputation','RF Imputation','KNN Imputation']
#x_labels = ['Full data','Mean Imputation','Zero Imputation','Median Imputation','Iterative Imputation','KNN Imputation']
x_labels = ['Full data','Mean Imputation','Zero Imputation','Median Imputation','Hot Deck Imputation','Iterative Imputation','KNN Imputation','Iterative Imputation1']
colors = ['r','g','b','y','m','orange','c','navy']
#colors = ['r','g','b','y','m','orange']

plt.figure(figsize=(12,6)) 
ax = plt.subplot(111)     

for i in np.arange(len(f1v)):
    ax.barh(i,f1v[i],color=colors[i],alpha=0.6,align='center')   
ax.set_title('Imputation Techniques with TBM Data: Validation set')
ax.set_xlim(left=np.min(f1v)*0.9,right=np.max(f1v)*1.1)  
ax.set_yticks(np.arange(len(f1v))) 
ax.set_xlabel('f1')        
ax.set_yticklabels(x_labels)  
plt.show()


plt.figure(figsize=(12,6)) 
ax = plt.subplot(111)     
for i in np.arange(len(f1t)):
    ax.barh(i,f1t[i],color=colors[i],alpha=0.6,align='center')   
ax.set_title('Imputation Techniques with TBM Data: Test set')
ax.set_xlim(left=np.min(f1t)*0.9,right=np.max(f1t)*1.1)  
ax.set_yticks(np.arange(len(f1t)))  
ax.set_xlabel('f1')
ax.set_yticklabels(x_labels)  
plt.show()








