#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# In[2]:




dataset = pd.read_csv('covid19.csv')


# In[3]:


dataset


# In[5]:



X = dataset.drop('Active Ratio', axis=1)  # Features
y = dataset['Active Ratio']  # Labels


# In[7]:


X


# In[9]:


y


# In[10]:


print("shape of data:", dataset.shape)


# In[11]:


dataset.head()


# In[12]:


dataset.tail()


# In[13]:


dataset.describe()


# In[14]:


dataset.dtypes


# In[15]:


dataset.isnull().sum()


# In[18]:



data1 = dataset.dropna()


data2 = dataset.fillna(value=0)


data3= dataset.interpolate()


# In[19]:


data1


# In[20]:


data2


# In[21]:


data3


# In[26]:


dataset['raviteja'] = dataset['Discharged'].astype('float')


# In[27]:


dataset['raviteja']


# In[28]:


data = dataset.drop_duplicates()


# In[29]:


data


# In[31]:


correlation_matrix = data.corr()


# In[32]:


correlation_matrix


# In[33]:


dataset['Discharge Ratio'].hist()
plt.show()


# In[34]:


plt.scatter(data['Active'], data['Discharged'])
plt.xlabel('Active')
plt.ylabel('Discharged')
plt.show()


# In[38]:


sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[46]:


x=dataset.drop(['State/UTs'],axis=1)


# In[47]:


x


# In[49]:


y=dataset['Discharge Ratio']


# In[50]:


y


# In[70]:


from sklearn.ensemble import RandomForestRegressor

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# In[75]:


regressor=RandomForestRegressor()


# In[76]:


regressor


# In[88]:


regressor.fit(x_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(x_test)


# In[89]:


y_pred


# In[100]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[101]:


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")


# In[ ]:




