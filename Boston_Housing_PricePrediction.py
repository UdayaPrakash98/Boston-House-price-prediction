#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


# In[9]:


bos=datasets.load_boston()
print(type(bos))
print(bos.keys())
print(bos.data.shape)
print(bos.feature_names)


# In[10]:


print(bos.DESCR)


# In[11]:


a=bos.feature_names
b=pd.DataFrame(bos.data,columns=a)
b['PRICE'] = bos.target
b.head(6)


# In[12]:


b.isnull().sum()


# In[13]:


print(b.describe())


# In[14]:


sns.set(rc={'figure.figsize':(15,8.27)})
plt.hist(b['PRICE'], bins=20)
plt.xlabel("House prices in $1000")
plt.show()#using matplotlib for normal distribution


# In[15]:


sns.displot( x=b["PRICE"],bins=10) #using seaborn for normal distribution


# In[16]:


#b1 = pd.DataFrame(bos.data, columns = bos.feature_names)
correlation_matrix = b.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True,)


# In[17]:


plt.figure(figsize=(10, 5))

features = ['LSTAT', 'RM']
target = b['PRICE']

for i, col in enumerate(features):
    plt.subplot(1,2, i+1)
    x = b[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title("Variation in House prices")
    plt.xlabel(col)
    plt.ylabel('"House prices in $1000"')
sns.jointplot(x=x,y=y,kind='reg')


# In[19]:


X_rooms = b.RM
Price = b.PRICE
X_rooms = np.array(X_rooms).reshape(-1,1)
y_price = np.array(Price).reshape(-1,1)
print(X_rooms.shape)
print(y_price.shape)


# In[20]:


X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_rooms, Price, test_size = 0.2, random_state=5)
print(X_train_1.shape)
print(X_test_1.shape)
print(Y_train_1.shape)
print(Y_test_1.shape)


# In[21]:


LR = LinearRegression()
LR.fit(X_train_1, Y_train_1)

y_predict_1 = LR.predict(X_train_1)
rmse = (np.sqrt(mean_squared_error(Y_train_1, y_predict_1)))
r2 = round(LR.score(X_train_1, Y_train_1),2)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


# In[22]:


y_pred_1 = LR.predict(X_test_1)
rmse = (np.sqrt(mean_squared_error(Y_test_1, y_pred_1)))
r2 = round(LR.score(X_test_1, Y_test_1),2)

print("The model performance for training set")
print("--------------------------------------")
print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(r2))
print("\n")


# In[23]:


prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1) 
plt.scatter(X_rooms,y_price)
plt.plot(prediction_space, LR.predict(prediction_space), color = 'r', linewidth = 2)
plt.ylabel('value of house/1000($)')
plt.xlabel('number of rooms')
plt.show()


# In[24]:


sns.regplot(x="RM", y="PRICE", data=b,color="g")


# In[25]:


X = b.drop('PRICE', axis = 1)
y = b['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=101)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)

# model evaluation for training set

y_train_predict = reg_all.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = round(reg_all.score(X_train, y_train),2)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


# In[26]:


y_pred = reg_all.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = round(reg_all.score(X_test, y_test),2)

print("The model performance for training set")
print("--------------------------------------")
print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(r2))
print("\n")


# In[259]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices ($1000)")
plt.ylabel("Predicted House Prices: ($1000)")
plt.xticks(range(0, int(max(y_test)),2))
plt.yticks(range(0, int(max(y_test)),2))
plt.title("Actual Prices vs Predicted prices")
prediction_space = np.linspace(min(y_test),max(y_pred))
plt.plot(prediction_space,color="g")


# In[28]:


sns.regplot(x=y_test, y=y_pred, data=b,color="g")


# In[ ]:


#used seaborn and matplotlib parrallel for different kind of underzstanding


# In[36]:


import pickle
import os


# In[38]:


os.chdir('F:\ML\ML assignmnets\ML Practical Assignmnets')
with open('Boston.pkl','wb') as file:
     pickle.dump(reg_all,file)


# In[43]:


with open('Boston.pkl','rb') as file:
     load=pickle.load(file)


# In[47]:


load.score(X_train,y_train)


# In[ ]:




