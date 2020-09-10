#import statements
import numpy as np
import pandas as pd
#import scipy.stats as stats
#import matplotlib.pyplot as plt
import sklearn
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Import boston dataset
from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)



# creating a boston dataframe
df= pd.DataFrame(boston.data,columns=boston.feature_names)

# appending price to it
df['Price'] = boston.target

corrmat = df.corr()
corrmat

# selecting features for the model
def getCorrelatedFeature(corrdata,threshold):
    feature = []
    value = []
    
    for i,index in enumerate(corrdata.index):
        if abs(corrdata[index])>threshold:
            feature.append(index)
            value.append(corrdata[index])
    
    df = pd.DataFrame(data = value,index=feature,columns=['Corr Value'])
    return df


threshold = 0.50
corr_Value = getCorrelatedFeature(corrmat['Price'],threshold)
correlated_data = df[corr_Value.index]

x = correlated_data.drop(labels=['Price'],axis=1)
y = correlated_data['Price']

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)


# Linear Rgresssion model
regressor = LinearRegression()
regressor.fit(X_train,y_train)

pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


#y_predict = model.predict(X_test)