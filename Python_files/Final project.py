#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object 

# In[ ]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[ ]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[ ]:


# notice: installing seaborn might takes a few minutes
# !conda install -c anaconda seaborn -y


# In[ ]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[ ]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[ ]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[ ]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[ ]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[ ]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[ ]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[ ]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[ ]:


X = Feature
X[0:5]


# What are our lables?

# In[ ]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[ ]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[ ]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_similarity_score

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# print(X_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

k = 4
Knn_neighbor = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
yhat = Knn_neighbor.predict(X_test)
b = jaccard_similarity_score(y_test, yhat)
print(f"Jaccard index score of the above classification method before finding best value of K : {b}\n")

# Finding besk K value
Mean_accuracy = []
Std_accuracy = []

for k in range(1,16):
      
    Knn_neighbor = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    yhat = Knn_neighbor.predict(X_test)
    
    
    Mean_accuracy.append(np.mean(yhat==y_test))
    
    Std_accuracy.append(np.std(yhat==y_test)/np.sqrt(yhat.shape[0]))
    
plt.plot(Mean_accuracy)
Highest_accuracy = max(Mean_accuracy)
K_value_of_max_accuracy = Mean_accuracy.index(max(Mean_accuracy)) + 1 #  since index starts from 0 addtion of 1 has to be done
print(Highest_accuracy)
print(K_value_of_max_accuracy)                                              


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
kNN_model = KNeighborsClassifier(n_neighbors=K_value_of_max_accuracy).fit(X_train,y_train)

#----------------------------------------------------------------------#



### Important Note : The below score computation is just for understanding not for evaluation ###

# yhat = Knn_neighbor.predict(X_test)

# a = f1_score(y_test, yhat, average='weighted')
# print(f"F1 score of the above classification method : {a}\n")

# b = jaccard_similarity_score(y_test, yhat)
# print(f"Jaccard index score of the above classification method : {b}\n")


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
import matplotlib.image as mpimg
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt


# In[ ]:


loanTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

loanTree.fit(X_train, y_train)


# In[ ]:



#----------------------------------------------------------------------#




### Important Note : The below score computation is just for understanding not for evaluation ###

# predTree = loanTree.predict(X_test)

# a = f1_score(y_test, predTree, average='weighted')
# print(f"F1 score of the above classification method : {a}\n")

# b = jaccard_similarity_score(y_test, predTree)
# print(f"Jaccard index score of the above classification method : {b}\n")


# # Support Vector Machine

# In[ ]:


from sklearn import svm

svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

#----------------------------------------------------------------------#





### Important Note : The below score computation is just for understanding not for evaluation ###

# svm_rbf_yhat = svm_rbf.predict(X_test)

# a = f1_score(y_test, svm_rbf_yhat, average='weighted')
# print(f"F1 score of the above classification method : {a}\n")

# b = jaccard_similarity_score(y_test, svm_rbf_yhat)
# print(f"Jaccard index score of the above classification method : {b}\n")

# svm_poly = svm.SVC(kernel='poly', degree = 9)
# svm_poly.fit(X_train, y_train)

# svm_poly_yhat = svm_poly.predict(X_test)

# a = f1_score(y_test, svm_poly_yhat, average='weighted')
# print(f"F1 score of the above classification method : {a}\n")

# b = jaccard_similarity_score(y_test, svm_poly_yhat)
# print(f"Jaccard index score of the above classification method : {b}\n")


# svm_linear = svm.SVC(kernel='linear')
# svm_linear.fit(X_train, y_train)

# svm_linear_yhat = svm_linear.predict(X_test)

# a = f1_score(y_test, svm_linear_yhat, average='weighted')
# print(f"F1 score of the above classification method : {a}\n")

# b = jaccard_similarity_score(y_test, svm_linear_yhat)
# print(f"Jaccard index score of the above classification method : {b}\n")

# svm_sigmoid = svm.SVC(kernel='sigmoid')
# svm_sigmoid.fit(X_train, y_train)

# svm_sigmoid_yhat = svm_sigmoid.predict(X_test)

# a = f1_score(y_test, svm_sigmoid_yhat, average='weighted')
# print(f"F1 score of the above classification method : {a}\n")

# b = jaccard_similarity_score(y_test, svm_sigmoid_yhat)
# print(f"Jaccard index score of the above classification method : {b}\n")


# In[ ]:





# In[ ]:





# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

sag_LR = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)

#----------------------------------------------------------------------#






### Important Note : The below score computation is just for understanding not for evaluation ###

# sag_LR_yhat = sag_LR.predict(X_test)

# sag_LR_yhat_prob = sag_LR.predict_proba(X_test)

# a = f1_score(y_test, sag_LR_yhat, average='weighted')
# print(f"F1 score of the above classification method : {a}\n")

# b = jaccard_similarity_score(y_test, sag_LR_yhat)
# print(f"Jaccard index score of the above classification method : {b}\n")

# c = (log_loss(y_test, sag_LR_yhat_prob))
# print(f"Log loss score of the above classification method : {c}\n")

# liblinear_LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

# liblinear_LR_yhat = liblinear_LR.predict(X_test)

# liblinear_LR_yhat_prob = liblinear_LR.predict_proba(X_test)

# a = f1_score(y_test, liblinear_LR_yhat, average='weighted')
# print(f"F1 score of the above classification method : {a}\n")

# b = jaccard_similarity_score(y_test, liblinear_LR_yhat)
# print(f"Jaccard index score of the above classification method : {b}\n")

# c = (log_loss(y_test, liblinear_LR_yhat_prob))
# print(f"Log loss score of the above classification method : {c}\n")

# newton_cg_LR = LogisticRegression(C=0.01, solver='newton-cg').fit(X_train,y_train)

# newton_cg_LR_yhat = newton_cg_LR.predict(X_test)

# newton_cg_LR_yhat_prob = newton_cg_LR.predict_proba(X_test)

# a = f1_score(y_test, newton_cg_LR_yhat, average='weighted')
# print(f"F1 score of the above classification method : {a}\n")

# b = jaccard_similarity_score(y_test, newton_cg_LR_yhat)
# print(f"Jaccard index score of the above classification method : {b}\n")

# c = (log_loss(y_test, newton_cg_LR_yhat_prob))
# print(f"Log loss score of the above classification method : {c}\n")

# lbfgs_LR = LogisticRegression(C=0.01, solver='lbfgs').fit(X_train,y_train)

# lbfgs_LR_yhat = lbfgs_LR.predict(X_test)

# lbfgs_LR_yhat_prob = lbfgs_LR.predict_proba(X_test)

# a = f1_score(y_test, lbfgs_LR_yhat, average='weighted')
# print(f"F1 score of the above classification method : {a}\n")

# b = jaccard_similarity_score(y_test, lbfgs_LR_yhat)
# print(f"Jaccard index score of the above classification method : {b}\n")

# c = (log_loss(y_test, lbfgs_LR_yhat_prob))
# print(f"Log loss score of the above classification method : {c}\n")

# saga_LR = LogisticRegression(C=0.01, solver='saga').fit(X_train,y_train)

# saga_LR_yhat = saga_LR.predict(X_test)

# saga_LR_yhat_prob = saga_LR.predict_proba(X_test)

# a = f1_score(y_test, saga_LR_yhat, average='weighted')
# print(f"F1 score of the above classification method : {a}\n")

# b = jaccard_similarity_score(y_test, saga_LR_yhat)
# print(f"Jaccard index score of the above classification method : {b}\n")

# c = (log_loss(y_test, saga_LR_yhat_prob))
# print(f"Log loss score of the above classification method : {c}\n")


# In[ ]:





# In[ ]:





# # Model Evaluation using Test set

# In[ ]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[ ]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[ ]:


test_df = pd.read_csv('loan_test.csv')

test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df.groupby(['education'])['loan_status'].value_counts(normalize=True)
test_df[['Principal','terms','age','Gender','education']].head()
Test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
Test_Feature = pd.concat([Test_Feature,pd.get_dummies(test_df['education'])], axis=1)
Test_Feature.drop(['Master or Above'], axis = 1,inplace=True)
Test_X = preprocessing.StandardScaler().fit(Test_Feature).transform(Test_Feature)
Test_y = test_df['loan_status'].values

# print(Test_X[0:3])
# print(Test_y[0:3])


# In[ ]:


Knn_testing = Knn_neighbor.predict(Test_X)
Knn_F1 = f1_score(Test_y, Knn_testing, average='weighted')
Knn_JI = jaccard_similarity_score(Test_y, Knn_testing)

Decision_tree_testing = loanTree.predict(Test_X)
DT_F1 = f1_score(Test_y, Decision_tree_testing, average='weighted')
DT_JI = jaccard_similarity_score(Test_y, Decision_tree_testing)

SVM_testing = svm_rbf.predict(Test_X)
SVM_F1 = f1_score(Test_y, SVM_testing, average='weighted')
SVM_JI = jaccard_similarity_score(Test_y, SVM_testing)

LogReg_testing = sag_LR.predict(Test_X)
LogReg_probability_testing = sag_LR.predict_proba(Test_X)
LogReg_F1 = f1_score(Test_y, LogReg_testing, average='weighted')
LogReg_JI = jaccard_similarity_score(Test_y, LogReg_testing)
LogReg_logloss = (log_loss(Test_y, LogReg_probability_testing))


# In[ ]:


scores={}
scores['Knn_F1'] = round(Knn_F1,4)
scores['Knn_JI'] = round(Knn_JI,4)
scores['DT_F1'] = round(DT_F1,4) 
scores['DT_JI'] = round(DT_JI,4)
scores['SVM_F1'] = round(SVM_F1,4) 
scores['SVM_JI'] = round(SVM_JI,4)
scores['LogReg_F1'] = round(LogReg_F1,4)
scores['LogReg_JI'] = round(LogReg_JI,4) 
scores['LogReg_logloss'] = round(LogReg_logloss,4)

print(scores)


# In[ ]:





# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# In[ ]:


print("| Algorithm          | Jaccard | F1-score | LogLoss |")
print("|--------------------|---------|----------|---------|")
print(f"| KNN                | {Knn_JI:.4f}  | {Knn_F1:.4f}   | NA      |")
print(f"| Decision Tree      | {DT_JI:.4f}  | {DT_F1:.4f}   | NA      |")
print(f"| SVM                | {SVM_JI:.4f}  | {SVM_F1:.4f}   | NA      |")
print(f"| LogisticRegression | {LogReg_JI:.4f}  | {LogReg_F1:.4f}   | {LogReg_logloss:.4f}  |")


# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
