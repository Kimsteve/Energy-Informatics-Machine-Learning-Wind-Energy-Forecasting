
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor



def linear_regression(x,y):
    print ("Training model: Linear Regression ")
    linear_regression = linear_model.LinearRegression()
    return linear_regression.fit(x,y)

# k = 1640
def k_nearest_neighbor(x,y):
    print ("Training model: k-Nearest Neighbor ")
    k_nearest_neighbor  = KNeighborsRegressor(n_neighbors=1440)
    return k_nearest_neighbor.fit(x,y)


#epsilon of 0.083
def supported_vector_regression(x,y):
    print ("Training model: Support Vector Regression ")
    supported_vector_regression = SVR(kernel='rbf', epsilon=0.083)
    return supported_vector_regression.fit(x,y.ravel())


#two hidden layers of 4 and 6 nodes respectively
def artificial_neural_networks(x,y):
    print ("Training model: Artificial Neural Network ")
    artificial_neural_networks = MLPRegressor(hidden_layer_sizes=(4,6))
    return artificial_neural_networks.fit(x,y.ravel())





