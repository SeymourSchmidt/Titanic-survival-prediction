
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import os


# In[3]:


os.getcwd()


# In[1]:


import pandas as pd
import numpy as np
from pandas import Series, DataFrame


# In[2]:


data_train = pd.read_csv('train.csv')


# In[3]:


data_train.head()


# In[5]:


data_train.info()


# In[16]:


import matplotlib.pyplot as plt


# In[30]:


fig = plt.figure()
fig.set(alpha=0.2)

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
df


# In[35]:


df = data_train


# In[47]:


df.head(3)


# In[90]:


df.Age.count()


# In[82]:


from sklearn.ensemble import RandomForestRegressor


# In[83]:


##replace none Age with RandomForestClassifier


# In[175]:


def set_missing_ages(df, N):
    
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp','Pclass']]
    
    #Group with age known& unkonwn
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    
    # set label:Age
    y = known_age[:,0]
    
    # set attributes
    X = known_age[:,1:]
    
    # fit datas to model
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000,n_jobs=-1)
    rfr.fit(X, y)
    
    if df.Age.count() < N:    
        
        # prediction
        predictedAges = rfr.predict(unknown_age[:, 1:])
    
        # fill with predition
        df.loc[(df.Age.isnull()),'Age'] = predictedAges
    
    return df,rfr

def set_missing_Fare(df, N):
    
    Fare_df = df[['Fare', 'Age', 'Parch', 'SibSp','Pclass']]
    
    #Group with age known& unkonwn
    known_Fare = Fare_df[Fare_df.Fare.notnull()].as_matrix()
    unknown_Fare = Fare_df[Fare_df.Fare.isnull()].as_matrix()
    
    # set label:Age
    y = known_Fare[:,0]
    
    # set attributes
    X = known_Fare[:,1:]
    
    # fit datas to model
    rfr_F = RandomForestRegressor(random_state=0, n_estimators=2000,n_jobs=-1)
    rfr_F.fit(X, y)
    
    if df.Fare.count() < N:    
        
        # prediction
        predictedFare = rfr2.predict(unknown_Fare[:, 1:])
    
        # fill with predition
        df.loc[(df.Fare.isnull()),Fare] = predictedFare
    
    return df,rfr2
        
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()),'Cabin'] = "No"
    return df

n= 891    
data_train, rfr = set_missing_ages(data_train, n)
    
data_train = set_Cabin_type(data_train)
data_train.head(5)


# In[ ]:


## discretization


# In[140]:


dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')


# In[141]:


## replace with new columns


# In[142]:


df =  pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)


# In[143]:


df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df.head(5)


# In[108]:


## scaling


# In[144]:


import sklearn.preprocessing as preprocessing
scaler = preprocessing.MinMaxScaler()
df['Age_scaled'] = scaler.fit_transform(df['Age'].as_matrix().reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].as_matrix().reshape(-1,1))
df.head(5)


# In[145]:


## Linear model building


# In[146]:


from sklearn import linear_model


# In[147]:


##  build new df and transfome in np


# In[148]:


train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()


# In[149]:


## train classifier LR


# In[150]:


y = train_np[:, 0]
X = train_np[:, 1:]


# In[151]:


clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
clf


# In[152]:


X.shape


# In[153]:


# Prediction


# In[154]:


## test set transform


# In[155]:


data_test = pd.read_csv('test.csv')
data_test.head(3)


# In[156]:


data_test.info()


# In[157]:


tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
tmp_df.head(5)


# ## fill null age

# In[159]:


null_age = tmp_df[tmp_df.Age.isnull()].as_matrix()


# In[166]:


null_Fare = tmp_df[tmp_df.Fare.isnull()].as_matrix()
X = null_Fare[:,1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()),'Age'] = predictedAges


# ## fill null fare

# In[173]:


tmp_df2 = data_test[['Fare', 'Age', 'Parch', 'SibSp', 'Pclass']]
tmp_df2.head(5)


# In[177]:


n= 891    
data_train, rfr2 = set_missing_ages(data_train, n)


# In[178]:


null_Fare = tmp_df2[tmp_df2.Fare.isnull()].as_matrix()
X = null_Fare[:,1:]
predictedFare = rfr2.predict(X)
data_test.loc[(data_test.Fare.isnull()),'Fare'] = predictedFare


# In[179]:


data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


# In[180]:


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].as_matrix().reshape(-1,1))
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].as_matrix().reshape(-1,1))
df_test.head(5)


# In[182]:


test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions.csv", index=False)
result.head


# In[183]:


pd.read_csv("logistic_regression_predictions.csv")

