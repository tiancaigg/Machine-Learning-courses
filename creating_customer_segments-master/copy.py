import numpy as np
import pandas as pd
import visuals as vs


try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True) #Return new object with labels in requested axis removed.
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

labels = data['Detergents_Paper']
new_data = data.copy()
new_data = data.drop(['Detergents_Paper'], axis = 1, inplace = True)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, labels, test_size=0.25, random_state=0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)

score = regressor.score(X_test, y_test)
print score