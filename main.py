############ importing the requirements module ############### 

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


########## loading the csv file to pandas dataframe ####### 

car_dataset = pd.read_csv(r'C:\Users\ramiu\Documents\Machine Learning Projects\Car Price Prediction\car data.csv')

###### inspecting the first five rows of the dataframe 

#print(car_dataset.head())

######### CHECKING THE NUMBER OF rows and coloumns #### 

print(car_dataset.shape)

####### getting some infirmation about the dataset 

print(car_dataset.info())

######## checking the number of missing value ####### 

print(car_dataset.isnull().sum())

########## checking the distribution of categorical data 

print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())


########## encoding the fuel type ########### 

car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

####### encoding seller type ######## 

car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

######## encoding the transmission coloumn ########## 

car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

print(car_dataset.head())

############ splitting the data into training data and test data ####

x = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
y = car_dataset['Selling_Price']
print(x)
print(y)

  ### splitting and test data ########### 

x_train , x_test , y_train  , y_test = train_test_split(x,y , test_size = 0.1, random_state=2)

######### model training ###### 

##### linear regression ######

##### loading the linear regression ###### 

lin_reg_model = LinearRegression()

lin_reg_model.fit(x_train,y_train)


##### model evaluation ############# 

training_data_prediction = lin_reg_model.predict(x_train)


###### r sqaure error ## 

error_score = metrics.r2_score(y_train,training_data_prediction)

print("R squared error:", error_score)

######## visualize the actual price and predicted price ####### 

plt.scatter(y_train,training_data_prediction)
plt.xlabel("Actual Price ")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

###### pedicted for test data ######### 

test_data_prediction = lin_reg_model.predict(x_test)

error_score = metrics.r2_score(y_test,test_data_prediction)

print("R squared error:",error_score)

plt.scatter(y_test,test_data_prediction)
plt.xlabel("Actual Price ")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()



############ lasso regression ######### 



lass_reg_model = Lasso()

lass_reg_model.fit(x_train,y_train)


##### model evaluation ############# 

training_data_prediction = lass_reg_model.predict(x_train)


###### r sqaure error ## 

error_score = metrics.r2_score(y_train,training_data_prediction)

print("R squared error:", error_score)

######## visualize the actual price and predicted price ####### 

plt.scatter(y_train,training_data_prediction)
plt.xlabel("Actual Price ")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

###### pedicted for test data ######### 

test_data_prediction = lass_reg_model.predict(x_test)

error_score = metrics.r2_score(y_test,test_data_prediction)

print("R squared error:",error_score)

plt.scatter(y_test,test_data_prediction)
plt.xlabel("Actual Price ")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()



