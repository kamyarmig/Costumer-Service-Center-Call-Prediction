
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



