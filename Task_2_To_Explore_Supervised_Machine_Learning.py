#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ***Task 2 - To Explore Supervised Machine Learning***

# In this regression task we will predict the percentage of
# marks that a student is expected to score based upon the
# number of hours they studied. This is a simple linear
# regression task as it involves just two variables.

# In[6]:


#importing the neccesary libraries
#Kailash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Check out the Data**

# In[4]:


data=pd.read_csv("student_scores.csv")


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.shape


# In[9]:


data.describe()


# EDA

# In[12]:


sns.pairplot(data)


# In[13]:


#By seeing the above scatter plots we can conclude that the marks and hours are positivly coorelated
sns.heatmap(data.corr())


# **Training a Linear Regression Model**
# 

# Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Marks.

# In[20]:


A= data.iloc[:, :-1].values  
B= data.iloc[:, 1].values 


# **Train Test Split**

# In[96]:


from sklearn.model_selection import train_test_split
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=0)


# **Creating and Training the Model**
# 
# ---
# 
# 

# In[97]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(A_train,B_train)


# **Model** **Evaluation**

# In[98]:


print(model.intercept_)
print(model.coef_)


# Interpreting the coefficients:
# 
# Holding all other features fixed, a 1 unit increase in Hours is associated with an increase of 9.73330705 Marks

# **Predictions** **from** **our** **Model**

# Let's grab predictions off our test set and see how well it did!

# In[99]:


predictions = model.predict(A_test)
line = model.coef_*A+model.intercept_


# In[100]:


plt.scatter(A, B)
plt.plot(A,line)
plt.show()


# In[101]:


plt.scatter(y_test,predictions)


# In[ ]:





# **Regression Evaluation Metrics**

# In[103]:


from sklearn import metrics


# In[104]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[104]:




