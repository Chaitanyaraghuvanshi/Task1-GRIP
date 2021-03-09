import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
from sklearn import metrics  


# Reading the dataset
url = "http://bit.ly/w-data"
dataset = pd.read_csv(url)
dataset.head()

x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:,1].values

# Plotting the distribution of scores
plt.scatter(x,y)  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()
print('The above plot shows high correlation between hours studied and percentage scored.')
#Splitting the data into training and testing set 
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.8, 
                            test_size=0.2, random_state=10) 

reg_model = LinearRegression()  
reg_model.fit(x_train, y_train) 

 # Predicting the scores
y_pred = regressor.predict(X_test)

#Plotting the predicted scores vs the testing scores
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,'r')
plt.legend(['Predicted Data','Testing Data'])
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')
plt.show()

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df) 

hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

#In order to evaluate the performance 
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 