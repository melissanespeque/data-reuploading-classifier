#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import warnings
warnings.filterwarnings('ignore') # Its for my shit macbook girls

#To find datasets
sys.path.append('../src') 
from datasets import get_iris_binary, plot_dataset

#Load and prepare the dataset (Setosa vs Versicolor)
X_train, X_test, y_train, y_test = get_iris_binary(task_type="setosa_vs_versicolor")

#Plot the 2D representation
plot_dataset(X_train, y_train)


# In[2]:


get_ipython().system('pip install qiskit qiskit-machine-learning pylatexenc')


# In[1]:


get_ipython().system('pip install "numpy<2"')


# In[ ]:




