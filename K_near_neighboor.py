# exercise 6.3.1


from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from numpy import cov


#Therefore the Classification y data is the the glass_type
y = glass_type
#The X data is the all of the features
X = Y2

#Shape of X matrix
N, M = X.shape
C = len(ClassNames)



# Plot the training data points (color-coded) and test data points.
figure(1)
Class_Colors = ['.r','.g', '.b','.c','.k','.m','.y']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,0], X_train[class_mask,1], Class_Colors[c])




## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# K-nearest neighbors
K_near=5

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=2
metric = 'minkowski'
metric_params = {} # no parameters needed for minkowski


# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=K_near, p=dist, 
                                    metric=metric, metric_params=metric_params)


errors = np.zeros((N,10))


k=0
for train_index, test_index in CV.split(X,y):

    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,10+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[k,l-1] = np.sum(y_est[0]!=y_test[0])


    # Plot the classfication results
    Class_Colors = ['or','og', 'ob','oc','ok','om','oy']
    for c in range(C):
        class_mask = (y_est==c)
        plot(X_test[class_mask,0], X_test[class_mask,1], Class_Colors[c], markersize=10)
        plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
    title('Synthetic data classification - KNN');

    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_est);
    accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
    figure(2);
    imshow(cm, cmap='binary', interpolation='None');
    colorbar()
    xticks(range(C)); yticks(range(C));
    xlabel('Predicted class'); ylabel('Actual class');
    title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));
    
    show()

print('Ran Exercise 6.3.1')