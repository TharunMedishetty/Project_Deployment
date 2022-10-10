#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud


# In[17]:


train=pd.read_csv("E:/THARUN/PROJECT/train.tsv",sep='\t')
test=pd.read_csv("E:/THARUN/PROJECT/test.tsv",sep='\t')


# In[18]:


train_size=len(train)
test_size=len(test)


# In[19]:


train_data=train.copy()
test_data=test.copy()


# In[20]:


combined_data = pd.concat([train_data,test_data])


# In[21]:


submission = test_data[['test_id']]


# In[22]:


combined_frac = combined_data.sample(frac=0.1).reset_index(drop=True)


# In[23]:


from string import punctuation
punctuation


# In[24]:


punctuation_symbols = []
for symbol in punctuation:
    punctuation_symbols.append((symbol, ''))


# In[25]:


import string
def remove_punctuation(sentence: str) -> str:
    return sentence.translate(str.maketrans('', '', string.punctuation))


# In[26]:


def remove_digits(x):
    x = ''.join([i for i in x if not i.isdigit()])
    return x


# In[27]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


# In[28]:


def remove_stop_words(x):
    x = ' '.join([i for i in x.lower().split(' ') if i not in STOPWORDS])
    return x


# In[29]:


def to_lower(x):
    return x.lower()


# In[30]:


def transform_category_name(category_name):
    try:
        main, sub1, sub2= category_name.split('/')
        return main, sub1, sub2
    except:
        return np.nan, np.nan, np.nan


# In[31]:


train_data['category_main'], train_data['subcat_1'], train_data['subcat_2'] = zip(*train_data['category_name'].apply(transform_category_name))
cat_train = train_data[['category_main','subcat_1','subcat_2', 'price']]


# In[32]:


combined_data.item_description = combined_data.item_description.astype(str)
descr = combined_data[['item_description', 'price']]
descr['count'] = descr['item_description'].apply(lambda x : len(str(x)))
descr['item_description'] = descr['item_description'].apply(remove_digits)
descr['item_description'] = descr['item_description'].apply(remove_punctuation)
descr['item_description'] = descr['item_description'].apply(remove_stop_words)
descr.head(20)


# In[33]:


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
descr['item_description'] = descr['item_description'].apply(porter.stem)
descr.tail(20)


# In[34]:


# Basic data imputation of missing values
def handle_missing_values(df):
  df['category_name'].fillna(value='missing',inplace=True)
  df['brand_name'].fillna(value='None',inplace=True)
  df['item_description'].fillna(value='None',inplace=True)


# In[35]:


def to_categorical(df):
    df['brand_name'] = df['brand_name'].astype('category')
    df['category_name'] = df['category_name'].astype('category')
    df['item_condition_id'] = df['item_condition_id'].astype('category')


# In[36]:


handle_missing_values(combined_frac)
to_categorical(combined_frac)


# In[37]:


handle_missing_values(combined_data)
to_categorical(combined_data)


# In[38]:


combined_frac.item_description = combined_frac.item_description.astype(str)
combined_frac['item_description'] = combined_frac['item_description'].apply(remove_digits)
combined_frac['item_description'] = combined_frac['item_description'].apply(remove_punctuation)
combined_frac['item_description'] = combined_frac['item_description'].apply(remove_stop_words)
combined_frac['item_description'] = combined_frac['item_description'].apply(to_lower)
combined_frac['name'] = combined_frac['name'].apply(remove_digits)
combined_frac['name'] = combined_frac['name'].apply(remove_punctuation)
combined_frac['name'] = combined_frac['name'].apply(remove_stop_words)
combined_frac['name'] = combined_frac['name'].apply(to_lower)
combined_frac.head()


# In[39]:


combined_data.item_description = combined_data.item_description.astype(str)
combined_data['item_description'] = combined_data['item_description'].apply(remove_digits)
combined_data['item_description'] = combined_data['item_description'].apply(remove_punctuation)
combined_data['item_description'] = combined_data['item_description'].apply(remove_stop_words)
combined_data['item_description'] = combined_data['item_description'].apply(to_lower)
combined_data['name'] = combined_data['name'].apply(remove_digits)
combined_data['name'] = combined_data['name'].apply(remove_punctuation)
combined_data['name'] = combined_data['name'].apply(remove_stop_words)
combined_data['name'] = combined_data['name'].apply(to_lower)
combined_data.head()


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer


# In[41]:


cv = CountVectorizer(min_df=10)
X_name = cv.fit_transform(combined_data['name'])


# In[42]:


cv = CountVectorizer()
X_category = cv.fit_transform(combined_data['category_name'])


# In[43]:


tv = TfidfVectorizer(max_features=55000, ngram_range=(1, 2), stop_words='english')
X_description = tv.fit_transform(combined_data['item_description'])


# In[44]:


lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(combined_data['brand_name'])


# In[45]:


from scipy.sparse import vstack, hstack, csr_matrix
X_dummies = csr_matrix(pd.get_dummies(combined_data[['item_condition_id', 'shipping']], sparse=True).values)


# In[46]:


sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()


# In[47]:


X_train_sparse = sparse_merge[:train_size]
X_test = sparse_merge[train_size:]


# In[48]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=12345)
y = np.log1p(train_data['price'])
i = 0;
for train_indicies, valid_indicies in kf.split(X_train_sparse):
    X_train, y_train = X_train_sparse[train_indicies], y[train_indicies]
    X_valid, y_valid = X_train_sparse[valid_indicies], y[valid_indicies]


# In[49]:


from sklearn.metrics import mean_squared_error, r2_score
def run_model_advText(model, X_train, y_train, X_valid, y_valid, verbose = False):
    model.fit(X_train, y_train)
    preds_valid = model.predict(X_valid)
    mse = mean_squared_error(y_valid,preds_valid)
    r_sq = r2_score(y_valid,preds_valid)
    print("Mean Squared Error Value : "+"{:.2f}".format(mse))
    print("R-Squared Value : "+"{:.2f}".format(r_sq))
    return model, mse, r_sq


# In[50]:


from sklearn import linear_model


# In[51]:


regression_model=linear_model.Ridge(solver = "saga", fit_intercept=False)
regression_model.fit(X_train, y_train)


# In[52]:


import pickle
DATA_PATH = "E:/THARUN/PROJECT/Model/" 
pickle_out = open(DATA_PATH+"Regression_Model.pkl", mode = "wb") 
pickle.dump(regression_model, pickle_out) 
#model = pickle.load(pickle_out)
pickle_out.close()


# In[ ]:




