#!/usr/bin/env python
# coding: utf-8

# # ` Analysis of the Advertising Media affecting Sales`
# 
# ~Submitted by `Nitesh sharma`   20114013  IIT Kanpur, India      
#  https://www.linkedin.com/in/niteshnit/

# ## Importing Libraries

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from matplotlib import pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


import statsmodels.api as sm


# In[6]:


import statsmodels.formula.api as smf


# In[7]:


from sklearn.linear_model import LinearRegression


# In[8]:


from sklearn.metrics import mean_squared_error,r2_score


# In[9]:


from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict


# In[10]:


from sklearn.decomposition import PCA


# In[11]:


from sklearn.cross_decomposition import PLSRegression,PLSSVD


# In[12]:


from sklearn.preprocessing import scale


# In[13]:


from sklearn import model_selection


# In[14]:


from sklearn.linear_model import Ridge,Lasso,ElasticNet


# In[15]:


from sklearn.linear_model import RidgeCV,LassoCV,ElasticNetCV


# ## Linear Methods for Regression

# # `Simple Linear Regression`

# We will understand the dataset first.

# In[16]:


ads = pd.read_csv("../input/advertising-dataset/advertising.csv")


# In[17]:


ads.head()


# In[18]:


ads = ads[['TV', 'Radio', 'Newspaper', 'Sales']]


# In[19]:


ads.rename(columns={"Radio": "radio", "Newspaper": "newspaper","Sales": "sales"},inplace=True)


# In[20]:


ads.head()


# In[21]:


ads.describe().T


# In[22]:


ads.dtypes


# In[23]:


ads.shape


# In[24]:


ads.isna().sum()


# Let's see correlations between variables.

# In[25]:


ads.corr()


# Let's see correlations in pairplot.

# In[26]:


g= sns.pairplot(ads,kind="reg",diag_kws= {'color': 'red'})

g.fig.suptitle("Correlation of Advertising Dataset", y=1.08)

plt.show()


# In[27]:


sns.jointplot(x="TV", y="sales",data=ads,kind="reg",color="green")

plt.show()


# Now we will create a simple linear regression model by using **statsmodel** library.

# In[28]:


X = ads.TV
X = sm.add_constant(X) # It will add a constant to X.
X.head()


# In[29]:


y = ads.sales # Dependent Variable -Target


# In[30]:


slr = sm.OLS(y,X) 


# In[31]:


model = slr.fit()


# In[32]:


model.summary()


# In[33]:


model.params # Main parameters


# In[34]:


model.summary().tables[1]


# In[35]:


model.conf_int()


# In[36]:


#Signifigant Level - P value

model.f_pvalue


# In[37]:


print("f_pvalue: ", "%.4f" % model.f_pvalue)


# In[38]:


model.fvalue


# In[39]:


model.tvalues


# In[40]:


#Sum of squares error of model

model.mse_model #This is very bad


# In[41]:


model.rsquared


# In[42]:


model.rsquared_adj


# In[43]:


#Predicted Values
model.fittedvalues[:5]


# In[44]:


#real values
y[:5]


# In[45]:


#Manual formula of our model
print("Sales: " , model.params[0] , "+ (TV*",model.params[1],")")


# In[46]:


ax = sns.regplot(ads["TV"],ads["sales"],ci=None,scatter_kws={"color":"purple"},marker="x")
ax.set_title("Sales:  7.03259 + TV*0.04753")
ax.set_ylabel("# of Sales")
ax.set_xlabel("Expenses for TV")

plt.show()


# Now we will create a model with **sklearn** library.

# In[47]:


X = ads[["TV"]]
sm.add_constant(X)
y = ads["sales"]


# In[48]:


lr = LinearRegression()
model = lr.fit(X,y)


# In[49]:


model.coef_.item()


# In[50]:


model.intercept_


# In[51]:


model.score(X,y) #R Squared


# In[52]:


model.predict(X)[:10]


# Let's predict a spesific value.

# In[53]:


model.predict([[20]])


# In[54]:


model.predict([[43],[20],[32]])


# #### Residuals

# In[55]:


slr = sm.OLS(y,X) 
model = slr.fit()
model.summary()


# In[56]:


mean_square = mean_squared_error(y,model.fittedvalues)
mean_square


# In[57]:


rmse = np.sqrt(mean_square)
rmse


# In[58]:


results = pd.DataFrame({"Real": y, "Prediction": model.predict(X),
                        "Residuals": y -(model.predict(X)),
                        "Square of Residuals": (y -(model.predict(X)))**2})


# In[59]:


results.head(10)


# In[60]:


np.sum(results["Square of Residuals"]) # root mean_square error


# In[61]:


np.mean(results["Square of Residuals"]) # mean_square error


# In[62]:


model.resid[:10]


# In[63]:


plt.plot(model.resid,c="r")
plt.title("Plot of Residuals")
plt.show()


# ## `Multiple Linear Regression`

# #### Model

# For a real world example, we will use *advertising* dataset.
# 
# It can be downloaded from here: https://www.kaggle.com/ashydv/advertising-dataset

# We will understand the dataset first.

# In[64]:


ads = pd.read_csv("../input/advertising-dataset/advertising.csv")
ads.rename(columns={"Radio": "radio", "Newspaper": "newspaper","Sales": "sales"},inplace=True)


# In[65]:


ads = ads[['TV', 'radio', 'newspaper', 'sales']]
ads.head()


# In[66]:


ads.shape


# Let's select all independent variables.

# In[67]:


X = ads.drop("sales",axis=1)
X.head()


# Now we will select dependent variable.

# In[68]:


y = ads["sales"]
y[:5]


# Now we will split our dataset as train and test set.

# In[69]:


ads.shape


# In[70]:


X_train = X.iloc[:160]
X_test = X.iloc[160:]
y_train = y[:160]
y_test = y[160:]


# In[71]:


print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_test.shape)


# First we will create our model with **statsmodel**.

# Rsquare is explanation rate. Results means we explain 0.98 of sales variable.

# In[72]:


mlr = sm.OLS(y_train,X_train)
model = mlr.fit()
model.summary()


# Now we will create our model with **sklearn**.

# In[73]:


mlr = LinearRegression()
model = mlr.fit(X_train,y_train)


# In[74]:


print("Intercept of Model-Bias: ",model.intercept_)
print("Coefficients of Model: ",model.coef_)


# In[75]:


print("Sales:", model.intercept_ ," + ",
      model.coef_[0],"* TV +",
      model.coef_[1],"* Radio +",
      model.coef_[2],"* Newspaper")


# #### Prediction

# Let's predict a spesific value.

# In[76]:


model.predict(pd.DataFrame([[35],[20],[45]]).T).item()


# In[77]:


train_root_mean_square = np.sqrt(mean_squared_error(y_train,model.predict(X_train)))


# In[78]:


print("root_mean_square_error of Training Set: ",train_root_mean_square)


# In[79]:


test_root_mean_square = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))


# In[80]:


print("root_mean_square_error of Test Set: ",test_root_mean_square)


# #### Model Tuning

# In[81]:


X = ads.drop("sales",axis=1)
y = ads["sales"]

X_train = X.iloc[:160]
X_test = X.iloc[160:]
y_train = y[:160]
y_test = y[160:]


# In[82]:


mlr = LinearRegression()
model = mlr.fit(X_train,y_train)


# In[83]:


np.sqrt(mean_squared_error(y_train,model.predict(X_train)))


# In[84]:


model.score(X_train,y_train)


# Let's calculate validated r2 score.

# In[85]:


cross_val_score(model,X,y,cv=10,scoring="r2").mean()


# In[86]:


#Train Rsquare
cross_val_score(model,X_train,y_train,cv=20,scoring="r2").mean()


# In[87]:


np.sqrt(-cross_val_score(model,X_train,y_train,cv=20,scoring="neg_mean_squared_error"))


# In[88]:


np.sqrt(-cross_val_score(model,X_train,y_train,cv=20,scoring="neg_mean_squared_error")).mean()


# In[89]:


#Test Rsquare
cross_val_score(model,X_test,y_test,cv=20,scoring="r2").mean()


# In[90]:


np.sqrt(-cross_val_score(model,X_test,y_test,cv=20,scoring="neg_mean_squared_error"))


# In[91]:


np.sqrt(-cross_val_score(model,X_test,y_test,cv=20,scoring="neg_mean_squared_error")).mean()


# ### Least-Squares Regression(Ordinary Least Squares)

# The most common method for fitting a regression line is the method of least-squares. This method calculates the best-fitting line for the observed data by minimizing the sum of the squares of the vertical deviations from each data point to the line (if a point lies on the fitted line exactly, then its vertical deviation is 0). Because the deviations are first squared, then summed, there are no cancellations between positive and negative values.
# 
# The Ordinary Least Squares procedure seeks to minimize the sum of the squared residuals. This means that given a regression line through the data we calculate the distance from each data point to the regression line, square it, and sum all of the squared errors together. This is the quantity that ordinary least squares seeks to minimize.

# ![image.png](attachment:image.png)

# This photo is taken by: https://miro.medium.com/max/2366/1*tQkyTR9yxDcS1GKVFhdQQA.jpeg

# ### Principal Component Analysis (PCA) 

# In[92]:


hts = pd.read_csv("../input/hitters/Hitters.csv")
hts.head()


# Now we will remove NA values.

# In[93]:


hts.dropna(inplace=True)


# In[94]:


hts.describe().T


# We will do **One Hot Encoding** to categorical columns.

# In[95]:


one_hot_encoded = pd.get_dummies(hts[["League","Division","NewLeague"]])
one_hot_encoded.head()


# In[96]:


new_hts = hts.drop(["League","Division","NewLeague","Salary"],axis=1).astype("float64")


# In[97]:


new_hts.head()


# In[98]:


X = pd.concat([new_hts,one_hot_encoded[["League_N","Division_W","NewLeague_N"]]],axis=1)
X.head()


# In[99]:


y = hts.Salary # Target-dependent variable


# Now we will split our dataset as train and test set.

# In[100]:


X.shape


# In[101]:


y.shape


# In[102]:


X_train = X.iloc[:200]
X_test = X.iloc[200:]
y_train = y[:200]
y_test = y[200:]

print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_test.shape)


# Let's do dimensionality reduction.

# In[103]:


pca = PCA()


# In[104]:


X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_test = pca.fit_transform(scale(X_test))


# In[105]:


X_reduced_train[:5]


# Let's see cumulative explanatioon rates.

# In[106]:


np.cumsum(np.round(pca.explained_variance_ratio_,decimals=3)*100)[:6]


# ### Principal Component Regression(PCR)

# In[107]:


hts = pd.read_csv("../input/hitters/Hitters.csv")
hts.head()


# Now we will remove NA values.

# In[108]:


hts.dropna(inplace=True)


# In[109]:


hts.describe().T


# We will do **One Hot Encoding** to categorical columns.

# In[110]:


one_hot_encoded = pd.get_dummies(hts[["League","Division","NewLeague"]])
one_hot_encoded.head()


# In[111]:


new_hts = hts.drop(["League","Division","NewLeague","Salary"],axis=1).astype("float64")


# In[112]:


X = pd.concat([new_hts,one_hot_encoded[["League_N","Division_W","NewLeague_N"]]],axis=1)
X.head()


# In[113]:


y = hts.Salary # Target-dependent variable


# Now we will split our dataset as train and test set.

# In[114]:


hts.shape


# In[115]:


#Independent Variables
X.shape


# In[116]:


#Dependent Variables
y.shape


# In[117]:


X_train = X.iloc[:210]
X_test = X.iloc[210:]
y_train = y[:210]
y_test = y[210:]

print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_test.shape)


# Let's do dimensionality reduction.

# In[118]:


pca = PCA()


# In[119]:


X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_test = pca.fit_transform(scale(X_test))


# Let's see cumulative explanatioon rates.

# In[120]:


np.cumsum(np.round(pca.explained_variance_ratio_,decimals=3)*100)[:6]


# Let's create a model.

# In[121]:


pcr = LinearRegression()
pcr_model = pcr.fit(X_reduced_train,y_train)


# In[122]:


print("Intercept: ",pcr_model.intercept_)
print("Coefficients: ",pcr_model.coef_)


# #### Prediction

# In[123]:


y_pred = pcr_model.predict(X_reduced_train)
y_pred[:10]


# In[124]:


#root mean square error for Train Set
np.sqrt(mean_squared_error(y_train,y_pred))


# In[125]:


#r2 score for Train Set
r2_score(y_train,y_pred)


# In[126]:


y_pred = pcr_model.predict(X_reduced_test)
y_pred[:10]


# In[127]:


#root mean square error for Test Set
np.sqrt(mean_squared_error(y_test,y_pred))


# In[128]:


#r2 score for Test Set
r2_score(y_test,y_pred)


# #### Model Tuning

# In[129]:


pcr = LinearRegression()
pcr_model = pcr.fit(X_reduced_train,y_train)
y_pred = pcr_model.predict(X_reduced_test)
print("Root mean square error: ",np.sqrt(mean_squared_error(y_test,y_pred)))


# Let's try different dimensions.

# In[130]:


pcr_model = pcr.fit(X_reduced_train[:,:3],y_train)
y_pred = pcr_model.predict(X_reduced_test[:,:3])
print("Root mean square error: ",np.sqrt(mean_squared_error(y_test,y_pred)))


# In[131]:


cross_val = model_selection.KFold(n_splits=7,
                                  shuffle=True,
                                  random_state=45)
pcr = LinearRegression()
Root_mean_sqaure_error = []


# In[132]:


X_reduced_train.shape


# In[133]:


for num in np.arange(X_reduced_train.shape[1]+1):
    score = np.sqrt(-1*model_selection.cross_val_score(pcr,X_reduced_train[:,:num],y_train.ravel(),
                                                       cv=cross_val,scoring="neg_mean_squared_error")).mean()
    
    Root_mean_sqaure_error.append(score)


# In[134]:


plt.plot(Root_mean_sqaure_error,"-v",c="r")
plt.xlabel("Component Numbers")
plt.ylabel("Root_mean_sqaure_error")
plt.title("PCR Model Tuning")

plt.show()


# Optimum value seems 6. Now we will create a model with 6 components.

# In[135]:


pcr = LinearRegression()
pcr_model = pcr.fit(X_reduced_train[:,:6],y_train)


# In[136]:


y_pred = pcr_model.predict(X_reduced_train[:,:6])
print("Root mean square error for Train set: ",np.sqrt(mean_squared_error(y_train,y_pred)))


# In[137]:


y_pred = pcr_model.predict(X_reduced_test[:,:6])
print("Root mean square error for Text set: ",np.sqrt(mean_squared_error(y_test,y_pred)))


# ### Shrinkage(Regularization) Methods

# #### `Partial Least Squares (PLS)`

# In[138]:


hts = pd.read_csv("../input/hitters/Hitters.csv")
hts.head()


# Now we will remove NA values.

# In[139]:


hts.dropna(inplace=True)


# In[140]:


hts.describe().T


# We will do **One Hot Encoding** to categorical columns.

# In[141]:


one_hot_encoded = pd.get_dummies(hts[["League","Division","NewLeague"]])
one_hot_encoded.head()


# In[142]:


new_hts = hts.drop(["League","Division","NewLeague","Salary"],axis=1).astype("float64")


# In[143]:


X = pd.concat([new_hts,one_hot_encoded[["League_N","Division_W","NewLeague_N"]]],axis=1)
X.head()


# In[144]:


y = hts.Salary # Target-dependent variable


# Now we will split our dataset as train and test set.

# In[145]:


hts.shape


# In[146]:


#Independent Variables
X.shape


# In[147]:


#Dependent Variables
y.shape


# In[148]:


X_train = X.iloc[:210]
X_test = X.iloc[210:]
y_train = y[:210]
y_test = y[210:]

print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_test.shape)


# In[149]:


pls_model = PLSRegression(n_components=7).fit(X_train,y_train)


# In[150]:


pls_model.coef_


# ##### Prediction

# In[151]:


pls_model


# In[152]:


X_train.head()


# In[153]:


pls_model.predict(X_train)[:10]


# In[154]:


y_pred=pls_model.predict(X_train)


# In[155]:


#Train Error
np.sqrt(mean_squared_error(y_train,y_pred))


# In[156]:


r2_score(y_train,y_pred)


# In[157]:


y_pred=pls_model.predict(X_test)


# In[158]:


#Test Error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[159]:


r2_score(y_test,y_pred)


# ##### Model Tuning

# In[160]:


pls_model


# In[161]:


cross_val = model_selection.KFold(n_splits=15,
                                  shuffle=True,
                                  random_state=45)
Root_mean_sqaure_error = []


# In[162]:


for num in np.arange(1,X_train.shape[1]+1):
    pls= PLSRegression(n_components=num)
    score = np.sqrt(-1*model_selection.cross_val_score(pls,X_train,y_train,
                                                       cv=cross_val,scoring="neg_mean_squared_error")).mean()
    
    Root_mean_sqaure_error.append(score)


# In[163]:


len(Root_mean_sqaure_error)


# In[164]:


X_train.shape[1]


# In[165]:


plt.plot(np.arange(1,X_train.shape[1]+1),np.array(Root_mean_sqaure_error),"-v",c="g")
plt.xlabel("Component Numbers")
plt.ylabel("Root_mean_sqaure_error")
plt.title("PLS Model Tuning")

plt.show()


# Now we will create our last model with optimal value that seems 8.

# In[166]:


pls_model = PLSRegression(n_components=8).fit(X_train,y_train)


# In[167]:


y_pred=pls_model.predict(X_train)


# In[168]:


#Train Error
np.sqrt(mean_squared_error(y_train,y_pred))


# In[169]:


r2_score(y_train,y_pred)


# In[170]:


y_pred=pls_model.predict(X_test)


# In[171]:


#Test Error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[172]:


r2_score(y_test,y_pred)


# ### ~`Ridge Regression ( L2 Regularization)`

# In[173]:


hts = pd.read_csv("../input/hitters/Hitters.csv")
hts.head()


# Now we will remove NA values.

# In[174]:


hts.dropna(inplace=True)


# In[175]:


hts.describe().T


# We will do **One Hot Encoding** to categorical columns.

# In[176]:


one_hot_encoded = pd.get_dummies(hts[["League","Division","NewLeague"]])
one_hot_encoded.head()


# In[177]:


new_hts = hts.drop(["League","Division","NewLeague","Salary"],axis=1).astype("float64")


# In[178]:


X = pd.concat([new_hts,one_hot_encoded[["League_N","Division_W","NewLeague_N"]]],axis=1)
X.head()


# In[179]:


y = hts.Salary # Target-dependent variable


# Now we will split our dataset as train and test set.

# In[180]:


hts.shape


# In[181]:


#Independent Variables
X.shape


# In[182]:


#Dependent Variables
y.shape


# In[183]:


X_train = X.iloc[:210]
X_test = X.iloc[210:]
y_train = y[:210]
y_test = y[210:]

print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_test.shape)


# In[184]:


ridge_model = Ridge(alpha=0.2).fit(X_train,y_train)


# In[185]:


ridge_model


# In[186]:


ridge_model.coef_


# In[187]:


lambda_values= 10**np.linspace(5,-2,150)*0.5
ridge_model = Ridge()
coefficients = []

for lam in lambda_values:
    ridge_model.set_params(alpha=lam)
    ridge_model.fit(X_train,y_train)
    coefficients.append(ridge_model.coef_)


# In[188]:


lambda_values[:10]


# In[189]:


coefficients[:3]


# In[190]:


ax = plt.gca()
ax.plot(lambda_values,coefficients)
ax.set_xscale("log")

plt.xlabel("Lambda Values")
plt.ylabel("Coefficients")
plt.title("Ridge Coefficients")
plt.show()


# ##### Prediction

# In[191]:


ridge_model


# In[192]:


ridge_model.predict(X_train)[:10]


# In[193]:


y_pred=ridge_model.predict(X_train)


# In[194]:


#Train Error
np.sqrt(mean_squared_error(y_train,y_pred))


# In[195]:


r2_score(y_train,y_pred)


# In[196]:


y_pred=ridge_model.predict(X_test)


# In[197]:


#Test Error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[198]:


r2_score(y_test,y_pred)


# ##### Model Tuning

# In[199]:


lambda_values= 10**np.linspace(5,-2,150)*0.5


# In[200]:


lambda_values[:10]


# In[201]:


Ridge_cv = RidgeCV(alphas=lambda_values,
                   scoring="neg_mean_squared_error",
                   normalize=True)


# In[202]:


Ridge_cv.fit(X_train,y_train)


# In[203]:


Ridge_cv.alpha_


# Now we will create our last model with optimal alpha value that seems 0.009568617603791272.

# In[204]:


ridge_tuned = Ridge(alpha=Ridge_cv.alpha_,normalize=True).fit(X_train,y_train)


# In[205]:


y_pred=ridge_tuned.predict(X_train)


# In[206]:


#Train Error
np.sqrt(mean_squared_error(y_train,y_pred))


# In[207]:


r2_score(y_train,y_pred)


# In[208]:


y_pred=ridge_tuned.predict(X_test)


# In[209]:


#Test Error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[210]:


r2_score(y_test,y_pred)


# ### ~`Lasso Regression( L1 Regularization)`

# ##### Model

# In[211]:


hts = pd.read_csv("../input/hitters/Hitters.csv")
hts.head()


# Now we will remove NA values.

# In[212]:


hts.dropna(inplace=True)


# In[213]:


hts.describe().T


# We will do **One Hot Encoding** to categorical columns.

# In[214]:


one_hot_encoded = pd.get_dummies(hts[["League","Division","NewLeague"]])
one_hot_encoded.head()


# In[215]:


new_hts = hts.drop(["League","Division","NewLeague","Salary"],axis=1).astype("float64")


# In[216]:


X = pd.concat([new_hts,one_hot_encoded[["League_N","Division_W","NewLeague_N"]]],axis=1)
X.head()


# In[217]:


y = hts.Salary # Target-dependent variable


# Now we will split our dataset as train and test set.

# In[218]:


hts.shape


# In[219]:


#Independent Variables
X.shape


# In[220]:


#Dependent Variables
y.shape


# In[221]:


X_train = X.iloc[:210]
X_test = X.iloc[210:]
y_train = y[:210]
y_test = y[210:]

print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_test.shape)


# In[222]:


lasso_model = Lasso(alpha=0.1).fit(X_train,y_train)


# In[223]:


lasso_model


# In[224]:


lasso_model.coef_


# In[225]:


lambda_values= 10**np.linspace(5,-2,150)*0.5
lasso_model = Lasso()
coefficients = []

for lam in lambda_values:
    lasso_model.set_params(alpha=lam)
    lasso_model.fit(X_train,y_train)
    coefficients.append(lasso_model.coef_)


# In[226]:


ax = plt.gca()
ax.plot(lambda_values*2,coefficients)
ax.set_xscale("log")

plt.axis("tight")
plt.xlabel("Lambda Values - Alpha")
plt.ylabel("Coefficients - Weights")
plt.title("Lasso Coefficients")
plt.show()


# ##### Prediction

# In[227]:


lasso_model


# In[228]:


lasso_model.predict(X_train)[:10]


# In[229]:


y_pred=lasso_model.predict(X_train)


# In[230]:


#Train Error
np.sqrt(mean_squared_error(y_train,y_pred))


# In[231]:


r2_score(y_train,y_pred)


# In[232]:


y_pred=lasso_model.predict(X_test)


# In[233]:


#Test Error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[234]:


r2_score(y_test,y_pred)


# ##### Model Tuning

# In[235]:


Lasso_cv = LassoCV(alphas=None,
                   cv=15,
                   max_iter=15000,
                   normalize=True)


# In[236]:


Lasso_cv.fit(X_train,y_train)


# In[237]:


Lasso_cv.alpha_


# Now we will create our last model with optimal alpha value that seems 0.07340278835886885.

# In[238]:


lasso_tuned = Lasso(alpha=Lasso_cv.alpha_).fit(X_train,y_train)


# In[239]:


y_pred=lasso_tuned.predict(X_train)


# In[240]:


#Train Error
np.sqrt(mean_squared_error(y_train,y_pred))


# In[241]:


r2_score(y_train,y_pred)


# In[242]:


y_pred=lasso_tuned.predict(X_test)


# In[243]:


#Test Error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[244]:


r2_score(y_test,y_pred)


# ### ~`Elastic Net Regression`

# In[245]:


hts = pd.read_csv("../input/hitters/Hitters.csv")
hts.head()


# Now we will remove NA values.

# In[246]:


hts.dropna(inplace=True)


# In[247]:


hts.describe().T


# We will do **One Hot Encoding** to categorical columns.

# In[248]:


one_hot_encoded = pd.get_dummies(hts[["League","Division","NewLeague"]])
one_hot_encoded.head()


# In[249]:


new_hts = hts.drop(["League","Division","NewLeague","Salary"],axis=1).astype("float64")


# In[250]:


X = pd.concat([new_hts,one_hot_encoded[["League_N","Division_W","NewLeague_N"]]],axis=1)
X.head()


# In[251]:


y = hts.Salary # Target-dependent variable


# Now we will split our dataset as train and test set.

# In[252]:


hts.shape


# In[253]:


#Independent Variables
X.shape


# In[254]:


#Dependent Variables
y.shape


# In[255]:


X_train = X.iloc[:210]
X_test = X.iloc[210:]
y_train = y[:210]
y_test = y[210:]

print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_test.shape)


# In[256]:


elastic_net_model = ElasticNet().fit(X_train,y_train)


# In[257]:


elastic_net_model


# In[258]:


elastic_net_model.coef_


# In[259]:


elastic_net_model.intercept_


# ##### Prediction

# In[260]:


elastic_net_model


# In[261]:


elastic_net_model.predict(X_train)[:10]


# In[262]:


y_pred=elastic_net_model.predict(X_train)


# In[263]:


#Train Error
np.sqrt(mean_squared_error(y_train,y_pred))


# In[264]:


r2_score(y_train,y_pred)


# In[265]:


y_pred=elastic_net_model.predict(X_test)


# In[266]:


#Test Error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[267]:


r2_score(y_test,y_pred)


# ##### Model Tuning

# In[268]:


elastic_net_cv = ElasticNetCV(cv=15,random_state=42)


# In[269]:


elastic_net_cv.fit(X_train,y_train)


# In[270]:


elastic_net_cv.alpha_


# Now we will create our last model with optimal alpha value that seems 1116.4729085556469.

# In[271]:


elastic_net_tuned = ElasticNet(alpha=elastic_net_cv.alpha_).fit(X_train,y_train)


# In[272]:


y_pred=elastic_net_tuned.predict(X_train)


# In[273]:


#Train Error
np.sqrt(mean_squared_error(y_train,y_pred))


# In[274]:


r2_score(y_train,y_pred)


# In[275]:


y_pred=elastic_net_tuned.predict(X_test)


# In[276]:


#Test Error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[277]:


r2_score(y_test,y_pred)


# ### Summary

# 
# 
# | Model | Train Error | Test Error |
# | --- | --- | --- |
# | Partial Least Squares(PLS) | 309.05 | 330.64 |
# | Ridge Regression(L2 Regularization) | 306.60 | 327.87 |
# | Lasso Regression(L1 Regularization) | 303.25 | 336.56 |
# | Elastic Net Regression | 326.91 | 328.92 |
# 
