#Getting the data
import pandas  as pd
dataframe=pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv")
#data analysis
print(dataframe.head(10))
print(dataframe.info())
print(dataframe.describe())
#data visulazation
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(9,9))
map1=sns.heatmap(dataframe.corr(),annot=True,cmap="coolwarm")
plt.show()
"""Here with correlation heatmap you can check which features have correlation more or less with eachother"""
sns.pairplot(dataframe,hue='logS')
plt.show()

#split dataframe
X=dataframe.drop('logS',axis=1)
y=dataframe['logS']
print("X-data ...")
print(X.head())
print("Y-data ...")
print(y.head())

#data splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)
print("Lengths ....")
print(len(x_train),len(x_test),len(y_train),len(y_test))

#Model building 

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor,
                              HistGradientBoostingRegressor)
#MODEL1
linear_regression=LinearRegression()
linear_regression.fit(x_train,y_train)
y_preds1=linear_regression.predict(x_test)
#MODEL2
random_forest=RandomForestRegressor(n_jobs=-1,max_depth=3)
random_forest.fit(x_train,y_train)
y_preds2=random_forest.predict(x_test)
#MODEL3
hist_grad=HistGradientBoostingRegressor(max_iter=400)
hist_grad.fit(x_train,y_train)
y_preds3=hist_grad.predict(x_test)


from sklearn.metrics import mean_squared_error,r2_score

mean_squared_error_linear_reg=mean_squared_error(y_preds1,y_test)
r2_score_linear_regression=r2_score(y_preds1,y_test)
mean_squared_error_rf=mean_squared_error(y_preds2,y_test)
r2_score_rf=r2_score(y_preds2,y_test)
mean_squared_error_hgbr=mean_squared_error(y_preds3,y_test)
r2_score_hgbr=r2_score(y_preds3,y_test)



import pandas as pd

print("CHECKING DIFFERENT MODELS ERRORS")


df = pd.DataFrame([
    ["Linear Regression", mean_squared_error_linear_reg, r2_score_linear_regression],
    ["Random Forest", mean_squared_error_rf, r2_score_rf],
    ["Hist Gradient Boosting Regressor", mean_squared_error_hgbr, r2_score_hgbr]
], columns=["Method", "Testing MSE", "Testing r2_score"])


print(df)