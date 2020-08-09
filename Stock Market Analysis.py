#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import Series, DataFrame
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')


# In[3]:


from pandas_datareader.data import DataReader


# In[4]:


from datetime import datetime


# In[5]:


from __future__ import division


# In[6]:


tech_list = ['AAPL','GOOG','MSFT','AMZN']


# In[7]:


end = datetime.now()
start = datetime(end.year-1,end.month,end.day)


# In[8]:


for stock in tech_list:

    globals()[stock] = DataReader(stock,'yahoo',start,end)


# In[9]:


AAPL.head()


# In[10]:


AAPL.describe()


# In[11]:


AAPL.info()


# In[12]:


AAPL['Adj Close'].plot(legend=True,figsize=(10,4)) 


# In[13]:


AAPL['Volume'].plot(legend=True,figsize=(10,4))


# In[14]:


ma_day = [10,20,50]

for ma in ma_day:
    
    column_name = "MA for %s days" %(str(ma))
    
    AAPL[column_name] = pd.Series(AAPL['Adj Close']).rolling(window=ma).mean()


# In[15]:


AAPL[['Adj Close','MA for 10 days', 'MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(10,4))


# In[16]:


AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

AAPL['Daily Return'].plot(figsize=(10,4),legend=True,linestyle='--',marker='o')


# In[17]:


sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color = 'purple')


# In[18]:


AAPL['Daily Return'].hist(bins=100)


# In[19]:


closing_df = DataReader(tech_list,'yahoo',start,end)['Adj Close']


# In[20]:


closing_df.head()


# In[21]:


tech_rets = closing_df.pct_change()


# In[22]:


tech_rets.head()


# In[23]:


sns.jointplot('GOOG','GOOG',tech_rets,kind='scatter',color='seagreen')


# In[24]:


sns.jointplot('GOOG','MSFT',tech_rets,kind='scatter')


# In[25]:


tech_rets.head()


# In[26]:


sns.pairplot(tech_rets.dropna())


# In[27]:


returns_fig = sns.PairGrid(tech_rets.dropna())

returns_fig.map_upper(plt.scatter, color='purple')

returns_fig.map_lower(sns.kdeplot, cmap='cool_d')

returns_fig.map_diag(plt.hist,bins=30)


# In[28]:


returns_fig = sns.PairGrid(closing_df)

returns_fig.map_upper(plt.scatter, color='purple')

returns_fig.map_lower(sns.kdeplot, cmap='cool_d')

returns_fig.map_diag(plt.hist,bins=30)


# In[29]:


sns.heatmap(tech_rets.dropna().corr(),square=True)


# In[30]:


corr = tech_rets.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, vmin=0, vmax=1,annot=True)


# In[31]:


sns.heatmap(closing_df.corr(),square=True)


# In[32]:


corr = closing_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, vmin=0, vmax=1,annot=True)


# In[33]:


#Risk Analysis
rets = tech_rets.dropna()


# In[34]:


area = np.pi*20

plt.scatter(rets.mean(),rets.std(),s = area)

plt.xlabel('Expected return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
    label,
    xy= (x, y), xytext = (50,50),
    textcoords = 'offset points', ha = 'right', va= 'bottom',
    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3',color='black'))


# In[35]:


#value at risk
#Bootstrap Method


# In[36]:


sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color = 'purple')


# In[37]:


rets.head()


# In[38]:


rets['AAPL'].quantile(0.05)


# In[39]:


#Value at risk
#Monte Carlo Method


# In[40]:


days = 365

dt = 1/days

mu = rets.mean()['GOOG']

sigma = rets.std()['GOOG']


# In[41]:


def stock_monte_carlo(start_price, days, mu, sigma):
    
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    
    for x in xrange(1, days):
        
        shock[x] = np.random.normal(loc=mu*dt,scale = sigma*np.sqrt(dt))
        
        drift[x] = mu * dt
        
        price[x]=price[x-1]+(price[x-1] * (drift[x]+shock[x]))
        
    return price    
    


# In[42]:


GOOG.head()


# In[44]:


start_price = 1175.040039

for run in xrange(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))
    
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')


# In[45]:


runs = 10000

simulations = np.zeros(runs)

for run in xrange(runs):
    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]


# In[46]:


q = np.percentile(simulations,1)

plt.hist(simulations, bins = 200)

#Starting Price
plt.figtext(0.6,0.8, s="Starting price: $%.2f" %start_price)

#Mean ending Price
plt.figtext(0.6,0.7, "Mean final price: $%.2f" %simulations.mean())

#Variance of the Price (within 99% confidence interval)
plt.figtext(0.6,0.6, "VaR(0.99): $%.2f" %(start_price - q))

#Display 1% quantile
plt.figtext(0.15,0.6, "q(0.99)): $%.2f" %q)

#Plot a line at 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

#Title
plt.title(u"Final Price distribution for Goggle stock after %s days" %days, weight= 'bold');


# In[ ]:




