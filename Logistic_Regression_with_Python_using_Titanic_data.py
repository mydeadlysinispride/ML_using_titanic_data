#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import numpy as np


# In[93]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[94]:


train = pd.read_csv("train.csv")


# In[95]:


train.head()


# In[96]:


train.count()


# In[97]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[98]:


sns.set_style('whitegrid')


# In[99]:


sns.countplot(x='Survived',hue='Sex',data=train)


# In[100]:


sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# In[101]:


sns.countplot(x='SibSp',data=train)


# In[102]:


train['Fare'].hist(bins=40,figsize=(20,8))


# In[103]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)


# In[104]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[105]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[106]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[107]:


train.drop('Cabin',axis=1,inplace=True)

train.head()
# In[108]:


sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[109]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[110]:


train = pd.concat([train,sex,embark],axis=1)


# In[111]:


train.head()


# In[112]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[113]:


train.drop(['PassengerId'],axis=1,inplace=True)


# In[114]:


train.head()


# In[128]:


X = train.drop(['Survived'],axis=1)
y = train['Survived']


# In[129]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[136]:


from sklearn.linear_model import LogisticRegression


# In[137]:


logmodel = LogisticRegression()


# In[138]:


logmodel.fit(X_train,y_train)


# In[139]:


predictions = logmodel.predict(X_test)


# In[140]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[141]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)


# In[ ]:




