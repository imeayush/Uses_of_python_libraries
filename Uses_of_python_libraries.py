#!/usr/bin/env python
# coding: utf-8

# # use of Numpy or numerical python 
# 

# In[1]:


# numpy it provide support for large multi-dimensional arrays and matrices alogng with a collection of mathematical function to operate on these arrays efficiently .
#numpy is a foundational package for verious scientific and engineering application involving numerical operation and data analsis


# In[2]:


import numpy as np 


# In[3]:


arr = np.array([[3,5,8,7,6,]])


# In[4]:


arr[0, 1]


# In[5]:


arr.dtype


# In[6]:


arr


# In[7]:


listarr = np.array ([[1,2,3,4],[5,6,7,8],[5,8,9,6]])


# In[8]:


listarr


# In[9]:


ide = np.identity(50)


# In[10]:


ide


# In[11]:


ide.shape


# In[12]:


arr = np.arange(100)


# In[13]:


arr


# In[14]:


x = [[1,2,3,4,5],[21,22,23,24],[60,61,62,63]]


# In[15]:


ar = np.array(x)


# In[16]:


ar


# In[17]:


ar.sum(axis=0)


# In[18]:


ar.flat


# In[19]:


for item in ar.flat:
    print(item)


# In[20]:


ar.ndim


# In[21]:


ar.size


# In[22]:


ar.nbytes


# In[23]:


ar2 = np.array([[1,2,3],[9,8,7],[3,5,9]])


# In[ ]:





# In[24]:


ar*ar2


# 

# # use of python pandas

# In[25]:


# the pandas is a powerful and popular open-source data manipulation and analysis library for python.
#it provides easy-to-use and highly efficient data structures,primarily the 'data frame and serise that ' enable to work with structures and labeled data devloped on the numpy ,pandas 
#pandas is well-suited for cleaning,exploring and analyzing tabular data 


# In[26]:


import pandas as pd 


# In[27]:


data202 = {
    "name" : ["alex","amexz","alexx","amaza"],
    "subject": [25,58,36,98],
    "place" : ["italy","canada","america","australia"]
}


# In[28]:


df = pd.DataFrame(data202)


# In[29]:


df


# In[30]:


newdf = pd.DataFrame(np.random.rand (380,6))


# In[31]:


newdf


# In[32]:


newdf.head()


# In[33]:


newdf.describe()


# In[34]:


newdf.min()


# In[35]:


newdf.max()


# In[36]:


newdf.info()


# In[37]:


newdf.head(10)


# In[38]:


type(newdf)


# In[39]:


newdf.describe()


# In[40]:


newdf[2][2] = "amexa"


# In[41]:


newdf[1][1] = "alex"


# In[42]:


newdf


# In[43]:


newdf.loc[[1,2,3,4,5,6,10],[1,3,4,5]]


# # use of python matplotlib 

# In[44]:


# matplotlib is a comprehensive 2D plotting library for python that generates high-quality visualization it provide a flexible and versatile platform for creating a wide range of static,animated and interactive plot and chart
#matplotlib is widely used in data visualization scientific research and other fields making it an essential tool for conveying complex data insights in a visually compelling manner. with a variety of customization option matplotlib 
#enables user to create publication-quality figure and visualization  


# In[45]:


from datetime import datetime, timedelta
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


import plotly.offline as pyoff
import plotly.graph_objs as go


# In[46]:


encoding = 'latin-1'


# In[47]:


filename = r'C:\Users\Ayush Lokhande\OnlineRetail.csv'


# In[48]:


tx_data = pd.read_csv('OnlineRetail.csv',encoding = "ISO-8859-1")
tx_data.head(5)


# In[49]:


# Convert 'InvoiceDate' column to datetime type
tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])

# Create 'InvoiceYearMonth' column
tx_data['InvoiceYearMonth'] = tx_data['InvoiceDate'].map(lambda date: 100*date.year + date.month)


# In[50]:


tx_data.head(10)


# In[51]:


tx_data['Revenue'] = tx_data['UnitPrice'] * tx_data['Quantity']


# In[52]:


tx_data.groupby('InvoiceYearMonth')['Revenue'].sum()


# In[53]:


tx_revenue = tx_data.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()


# In[54]:


tx_revenue


# In[55]:


plot_data = [
    go.Scatter(
    x=tx_revenue['InvoiceYearMonth'],
    y=tx_revenue['Revenue'],
    )
]

plot_layout = go.Layout(
        xaxis = {"type": "category"},
    title='Montly Revenue'
)


# In[56]:


fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[57]:


tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)


# In[58]:


tx_uk.head()


# In[59]:


tx_monthly_active = tx_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()


# In[60]:


tx_monthly_active


# In[61]:


plot_data = [
    go.Bar(
        x = tx_monthly_active['InvoiceYearMonth'],
        y = tx_monthly_active['CustomerID'],
    )
]

plot_layout = go.Layout(
         xaxis = {"type": "category"},
         title = 'Monthly Active Customers')


# In[62]:


fig = go.Figure(data = plot_data, layout=plot_layout)
pyoff.iplot(fig)


# # use of python tensorflow :-
# 

# In[63]:


# this librarie is open source libraries this can devloped by google brain team ,
# it helps the devloper desing building and train deeplearning model
#in the graph we use nodes to represent the mathametical operation and is to represent the data that is communicate from 1 node to anothe node
#It has become one of the most widely used frameworks in the field of artificial intelligence and machine learning


# In[64]:


import tensorflow as tf
from tensorflow import keras 
tf.__version__


# In[67]:


NB_CLASSES = 10
RESHAPED = 784


# In[69]:


model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(NB_CLASSES,
                            input_shape=(RESHAPED,),
                            kernel_initializer = 'zeros',
                            name = 'dense_layer',
                            activation = 'softmax'))


# In[70]:


model.summary()


# In[72]:


#Sequential()Model
from tensorflow.keras.models import Sequential
model = Sequential()
#add()
from tensorflow.keras.layers import Dense
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))


# In[73]:


model.compile(loss = 'categorical_crossentropy',
             optimizer = "sgd",
             metrics=['accuracy'])


# In[76]:


x_train = ["abc","abc","abc","abc","abc","abc","abc"]
y_train=[1,2,3,4,5,6,7]


# In[78]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:





# In[ ]:





# In[85]:


model.fit(x_train, y_train, epochs=5, batch_size=32)


# In[82]:


loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)


# # use of python PyTorch 

# In[86]:


# PyTorch is a popular open-source deeplearning library in python that facilitates the developemnt and training of neural networks. known for its dynamic computational graph
# and eager execution ,PYTorch offers flexibility and ease of use .The library provides a versatile tensor data structure an intutive neural network module and automatic differentition through its autograd system 
# PYTorch supports GPU acceleration,enabling efficient training on graphics processing unit with a vibrant choice for reserchers snd practitioners in the machine learning and ai 


# In[4]:


import torch


# In[3]:


get_ipython().system('pip install torch torchvision torchaudio')


# In[5]:


import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple dataset
X = torch.rand((100, 1))  # 100 samples with 1 feature
y = torch.randint(0, 2, (100,))  # Binary labels

# Define a simple neural network
model = nn.Sequential(
    nn.Linear(1, 10),  # Input: 1 feature, Output: 10 neurons
    nn.ReLU(),
    nn.Linear(10, 1),  # Input: 10 neurons, Output: 1 neuron (binary classification)
    nn.Sigmoid()  # Sigmoid activation for binary classification
)

# Define loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)

    # Calculate loss
    loss = criterion(y_pred.squeeze(), y.float())

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the trained model
with torch.no_grad():
    test_X = torch.rand((10, 1))  # 10 test samples
    predictions = model(test_X).squeeze().round().int()
    print(f'Test Predictions: {predictions}')


# In[ ]:




