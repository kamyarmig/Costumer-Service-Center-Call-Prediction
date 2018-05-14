
import numpy as np
import pandas as pd

from datetime import datetime
import warnings
import calendar
#import rpy2
#import rpy2.robjects.packages as packages
#import rpy2.robjects.lib.ggplot2 as ggplot2
# fix random seed for reproducibility
np.random.seed(7)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# # __Reading data__



train_data_e = pd.read_csv("./Trainindata.csv")  #reading data for data summary and exploratory data analysis
train_data = pd.read_csv("./Trainindata.csv")  #reading data for prediction


# # Data Summary and Exploratory Data Analysis


train_data_e.head()


# Here we split date, month, weekday and hour:



train_data_e["date"] = train_data_e.datetime.apply(lambda x : x.split()[0])
train_data_e["year"] = train_data_e.datetime.apply(lambda x : x.split()[0].split("/")[2])
train_data_e["month"] = train_data_e.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%m/%d/%Y").month])
train_data_e["weekday"] = train_data_e.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%m/%d/%Y").weekday()])
train_data_e["hour"] = train_data_e.datetime.apply(lambda x : x.split()[1].split(":")[0])
train_data_e=train_data_e.drop(["date"],axis=1)  
train_data_e.head()


# We change seasons from numbers to their actual names:



train_data_e["season"] = train_data_e.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })
train_data_e.head()


# We change weather from numbers to their actual definition:



train_data_e["weather"] = train_data_e.weather.map({1: " Clear",                                                    2 : " Cloudy",                                                     3 : " Light Rain",                                                     4 :" Heavy Rain" })
train_data_e.head()




train_data_e.dtypes


# Here we change the type of some variables to catagorical:



list_of_cat_vars = ["holiday","workingday","season","weather","year","month","weekday","hour"]
for var_index in list_of_cat_vars:
    train_data_e[var_index] = train_data_e[var_index].astype("category")
    train_data_e.head()
    train_data_e.dtypes
    train_data_e = train_data_e.reset_index()
train_data_e = train_data_e.rename(columns = {'index':'time_stamp'})
train_data_e  = train_data_e.drop(["datetime"],axis=1)    
train_data_e.head()
msno.bar(train_data_e,figsize=(12,5))

sns.set_style("whitegrid")

fig,(ax1) = plt.subplots(ncols=1)
fig.set_size_inches(12, 5)
sns.regplot(x="time_stamp", y="Calls", data=train_data_e,ax=ax1, fit_reg=False)
ax1.set( title='Number of Calls in our training dataset',label='big')


# Histogram:

# As we can see there are quite a few outliers, we plot the histogram and probability plot which shows the distribution of Calls against normal distribution in the following:

# In[103]:


fig,(ax1) = plt.subplots(ncols=1)
fig.set_size_inches(12, 5)
sns.distplot( train_data_e["Calls"],ax=ax1,kde=False)
ax1.set( ylabel='# of Calls',title='Number of Calls in our training dataset',label='big')


# In[104]:


fig,axes = plt.subplots(nrows=2)
fig.set_size_inches(10, 10)
sns.distplot(train_data_e["Calls"],ax=axes[0])
axes[0].set( ylabel='Probability',title="Probability of Calls",label='big')
stats.probplot(train_data_e["Calls"], dist=stats.norm,  fit=True, plot=axes[1])


# It can be seen our response variable (Calls) is skewed toward right and the distribution is like log-normal distribution. Moreover, we have quite a few number of outliers. Possibly, we can correct for this for taking log transformation:

# In[105]:


fig,axes = plt.subplots(nrows=2)
fig.set_size_inches(10, 10)
sns.distplot(np.log(train_data_e["Calls"]),ax=axes[0])
axes[0].set( ylabel='Probability',title="Probability of log of Calls",label='big')
stats.probplot(np.log(train_data_e["Calls"]), dist=stats.norm,  fit=True, plot=axes[1])


# It can be seen our "Calls" is much closer to normal distribution after log transformation. One can also take out outliers, but we don't do that here.

# __Box Plots__

# In[106]:


fig, axes = plt.subplots(nrows=3,ncols=2)
fig.set_size_inches(15, 15)
sns.boxplot(data=train_data_e,y="Calls",ax=axes[0][0])
axes[0][0].set(ylabel='Calls',title="Calls")

sns.boxplot(data=train_data_e,y="Calls",x="hour",ax=axes[0][1])
axes[0][1].set(xlabel='Hour Of The Day', ylabel='Call',title="Calls Across Hour Of The Day")

sns.boxplot(data=train_data_e,y="Calls",x="season",ax=axes[1][0])
axes[1][0].set(xlabel='Season', ylabel='Call',title="Calls Across Season")

sns.boxplot(data=train_data_e,y="Calls",x="workingday",ax=axes[1][1])
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Calls Across Working Day")

sns.boxplot(data=train_data_e,y="Calls",x="weather",ax=axes[2][0])
axes[2][0].set(xlabel='Weather', ylabel='Count',title="Calls Across Weather")


sns.boxplot(data=train_data_e,y="Calls",x="weekday",ax=axes[2][1])
axes[2][1].set(xlabel='Week Day', ylabel='Count',title="Calls Across Weather")


# In the upper left graph, we can see there are a lot of outliers for "Calls" which results in skewness toward right as we saw in the previous graph.
# 
# In Hour of the Day boxplot, it can be seen that the median values are higher at around 8 am and 5 pm which can be explained by waking up and coming back from work. Moreover, there are some very big outliers in 5 and 6 pm. 
# 
# By seeing seasons, we can see Spring has relatively lower Calls, and Summer has relatively higher Calls which can be due to the higher consumption of electricity in Summer.
# 
# We can see most of outliers are related to "working days", however, the outliers from "non-working days" have higher values.
# 
# Moreover, we can see there are much more outliers in "Light Rain" compared with other weather situations.
# 
# Finally,  we can see there are relatively fewer calls during weekends, and we have higher outliers in Thursdays. 

# __Data Visualization__



fig,(ax1,ax2)= plt.subplots(nrows=2)

fig.set_size_inches(12,15)
sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]
hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]


hourAggregated = pd.DataFrame(train_data_e.groupby(["hour","weekday"],sort=True)["Calls"].mean()).reset_index()
sns.pointplot(x=hourAggregated["hour"], y=hourAggregated["Calls"],hue=hourAggregated["weekday"],hue_order=hueOrder, data=hourAggregated, join=True,ax=ax1)
ax1.set(xlabel='Hour Of The Day', ylabel='Average Calls',title="Average Calls By Hour Of The Day Across Weekdays",label='big')


hourAggregated = pd.DataFrame(train_data_e.groupby(["hour","season"],sort=True)["Calls"].mean()).reset_index()
sns.pointplot(x=hourAggregated["hour"], y=hourAggregated["Calls"],hue=hourAggregated["season"], data=hourAggregated, join=True,ax=ax2)
ax2.set(xlabel='Hour Of The Day', ylabel='Average Calls',title="Average Calls By Hour Of The Day Across Season",label='big')


#monthAggregated = pd.DataFrame(train_data_e.groupby("month")["Calls"].mean()).reset_index()
#monthSorted = monthAggregated.sort_values(by="Calls",ascending=False)
#sns.barplot(data=monthSorted,x="month",y="Calls",ax=ax3,order=sortOrder)
#ax3.set(xlabel='Month', ylabel='Avearage Calls',title="Average Calls By Month")


#monthAggregated = pd.DataFrame(train_data_e.groupby("month")["windspeed"].mean()).reset_index()
#monthSorted = monthAggregated.sort_values(by="windspeed",ascending=False)
#sns.barplot(data=monthSorted,x="month",y="windspeed",ax=ax4,order=sortOrder)
#ax4.set(xlabel='Month', ylabel='Avearage Calls',title="Average Calls By Month")


# We can see there are two peaks in each day: one around 8 am and the other one around 5-6 pm which are probably due to wake up and coming back from work. The evening peak is slightly higher in Thursdays.
# 
# We see the same pattern of peaks in different seasons, summer has the highest peak and spring has the lowest one.
# 
# 

# __Correlations__

# We use the actual data (before transforming catargorical variables to their names) and calculate correlation matrix:

train_data["date"] = train_data.datetime.apply(lambda x : x.split()[0])
train_data["month"] = train_data.date.apply(lambda dateString : datetime.strptime(dateString,"%m/%d/%Y").month)
train_data["weekday"] = train_data.date.apply(lambda dateString : datetime.strptime(dateString,"%m/%d/%Y").weekday())
train_data["hour"] = train_data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")


# In[109]:


cor_mat=train_data.corr()
cor_mat


# In[110]:


mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(cor_mat, mask=mask,vmax=.8, square=True,annot=True)


# We can see windspeed, temp and humidity has the highest correlation with number of Calls.

# We dropped date datetime and date:

# In[111]:


features_to_drop=["datetime","date"]
train_data  = train_data.drop(features_to_drop,axis=1)
train_data.head()


# We read test dataset and prepare it:

# In[166]:



test_data = pd.read_csv("./Testdata_3.csv")
test_data["date"] = test_data.datetime.apply(lambda x : x.split()[0])
test_data["month"] = test_data.date.apply(lambda dateString : datetime.strptime(dateString,"%m/%d/%Y").month)
test_data["weekday"] = test_data.date.apply(lambda dateString : datetime.strptime(dateString,"%m/%d/%Y").weekday())
test_data["hour"] = test_data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")
test_data  = test_data.drop(features_to_drop,axis=1)


# Making test and training datasets:

# In[167]:


ytrain=train_data.ix[:,train_data.columns=='Calls']
Xtrain=train_data.ix[:,train_data.columns != 'Calls']

ytest=test_data.ix[:,test_data.columns=='Calls']
Xtest=test_data.ix[:,test_data.columns != 'Calls']


# Since we have catagorical data, here we used OneHotEncoder to encode our datasets:

# In[168]:


from sklearn.preprocessing import OneHotEncoder 

encoder = OneHotEncoder()
Xtrain_cat=Xtrain[["holiday","workingday","season","weather","month","weekday","hour"]]
Xtrain_num=Xtrain[['temp','humidity','windspeed']]
Xtrain_cat_enc=encoder.fit_transform(Xtrain_cat).toarray()
Xtrain_en=np.concatenate((Xtrain_num, Xtrain_cat_enc), axis = 1)

Xtest_cat=Xtest[["holiday","workingday","season","weather","month","weekday","hour"]]
Xtest_num=Xtest[['temp','humidity','windspeed']]
Xtest_cat_enc=encoder.fit_transform(Xtest_cat).toarray()
Xtest_en=np.concatenate((Xtest_num, Xtest_cat_enc), axis = 1)


# __Linear Regression__

# In[169]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
#warnings.filterwarnings("ignore", category=DeprecationWarning)


ols = LinearRegression()

#making log variables
ytrain_log = np.log1p(ytrain)
ytest_log = np.log1p(ytest)

#train the model
ols.fit(X = Xtrain,y = ytrain_log) 

# Make predictions
ols_preds = ols.predict(X= Xtest)


# R squared:

# In[170]:


print "R squared Value For Linear Regression: ",metrics.r2_score(ytest_log, ols_preds)


# Mean Squared Error:

# In[171]:


print "MSRE Value For Linear Regression: ",metrics.mean_squared_error(ytest_log,ols_preds)


# Here, we run the same model with OneHotEncode since these encoding scheme usually do better in different algorithms:

# In[173]:


ols.fit(X = Xtrain_en,y = ytrain_log) 
ols_preds = ols.predict(X= Xtest_en)

print "R squared Value For Linear Regression: ",metrics.r2_score(ytest_log, ols_preds)
print "MSRE Value For Linear Regression: ",metrics.mean_squared_error(ytest_log,ols_preds)


# We can see the model improved significantly with OneHotEncode.

# __Regulariztion model-Ridge__

# We use Ridge regression and we also hyper-tune ridge parameter and then fit and predict:
# In[174]:


ridge_model = Ridge()
#parameters
ridge_params_ = { 'alpha':[1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6]}

#finding the best parameters based on RMSE and 5-fold CV
grid_ridge_m = GridSearchCV( ridge_model,
                          ridge_params_,
                          scoring = 'mean_squared_error',
                          cv=5)

grid_ridge_m_f=grid_ridge_m.fit( Xtrain, ytrain_log )
ridge_preds = grid_ridge_m.predict(X= Xtest)
print (grid_ridge_m.best_params_)


# The optimal parameter is 10,000

# In[175]:


print "R squared Value For Linear Regression: ",metrics.r2_score(ytest_log, ridge_preds)
print "RMSE Value For Linear Regression: ",metrics.mean_squared_error(ytest_log,ridge_preds)


# We can plot RMSE vs. ridge parameter and see the optimal value:

# In[176]:


fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_ridge_m.grid_scores_)
df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])
df["rmse"] = df["mean_validation_score"].apply(lambda x:-x)
sns.pointplot(data=df,x="alpha",y="rmse",ax=ax)
ax.set(title="RMSE vs. Alpha in Ridge Regression",label='big')


# Again, we do Ridge regression on OneHotEncoded data:

# In[177]:


#finding the best parameters based on RMSE and 5-fold CV
grid_ridge_m = GridSearchCV( ridge_model,
                          ridge_params_,
                          scoring = 'mean_squared_error',
                          cv=5)

grid_ridge_m.fit( Xtrain_en, ytrain_log )
ridge_preds = grid_ridge_m.predict(X= Xtest_en)
print (grid_ridge_m.best_params_)


# In[178]:


print "R squared Value For Linear Regression: ",metrics.r2_score(ytest_log, ridge_preds)
print "RMSE Value For Linear Regression: ",metrics.mean_squared_error(ytest_log,ridge_preds)


# Again, We see the model performs much better with OneHotEncode. The following graph also shows Ridge regression parameter vs. RMSE.

# In[179]:


fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_ridge_m.grid_scores_)
df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])
df["rmse"] = df["mean_validation_score"].apply(lambda x:-x)
sns.pointplot(data=df,x="alpha",y="rmse",ax=ax)
ax.set(title="RMSE vs. Alpha in Ridge Regression with OneHotEncode",label='big')


# __Lasso Regression__

# In[180]:


lasso_model = Lasso()

alpha  = 1/np.array([1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6])
lasso_params_ = { 'alpha':alpha}

grid_lasso_m = GridSearchCV( lasso_model,
                            lasso_params_,
                            scoring ='mean_squared_error',
                            n_jobs=-1,
                            cv=5)

grid_lasso_m.fit( Xtrain, ytrain_log )
lasso_preds = grid_lasso_m.predict(X= Xtest)
print "Optimal parameter: ",grid_lasso_m.best_params_
print "R squared Value For Linear Regression: ",metrics.r2_score(ytest_log, ridge_preds)
print "RMSE Value For Linear Regression: ",metrics.mean_squared_error(ytest_log,ridge_preds)

fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_lasso_m.grid_scores_)
df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])
df["rmse"] = df["mean_validation_score"].apply(lambda x:-x)
sns.pointplot(data=df,x="alpha",y="rmse",ax=ax)
ax.set(title="RMSE vs. Alpha in LASSO Regression",label='big')


# with OneHotEncode:

# In[181]:


grid_lasso_m = GridSearchCV( lasso_model,
                            lasso_params_,
                            scoring ='mean_squared_error',
                            n_jobs=-1,
                            cv=5)

grid_lasso_m.fit( Xtrain_en, ytrain_log )
lasso_preds = grid_lasso_m.predict(X= Xtest_en)
print "Optimal parameter: ",grid_lasso_m.best_params_
print "R squared Value For Linear Regression: ",metrics.r2_score(ytest_log, ridge_preds)
print "RMSE Value For Linear Regression: ",metrics.mean_squared_error(ytest_log,ridge_preds)

fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_lasso_m.grid_scores_)
df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])
df["rmse"] = df["mean_validation_score"].apply(lambda x:-x)
sns.pointplot(data=df,x="alpha",y="rmse",ax=ax)
ax.set(title="RMSE vs. Alpha in LASSO Regression with OneHotEncode",label='big')
