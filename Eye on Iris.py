# Written by Allyn Zheng on Jan 13, 2019
# A program to classify an iris flower based on its petal and sepal lengths,
#   considered the "Hello World" of machine learning.
# Training and testing data provided from UCI machine learning library: 
#   https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
# Inspired and guided by Jason Brownlee's "Machine Learning Mastery":
#   https://machinelearningmastery.com/machine-learning-with-python/

# IMPORT LIBRARIES

print("STEP 1: IMPORT LIBRARIES")
from matplotlib import pyplot
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# LOAD CSV DATA

print("STEP 2: LOAD CSV DATA")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/\
iris.data"
titles = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataframe = read_csv(url, names = titles)

# DATASET SUMMARY

print("STEP 3: DATASET SUMMARY")
print(dataframe.shape) # to double-check number of columns and rows
print(dataframe.head(10)) # to peek at the first 10 rows of data
print(dataframe.describe()) # to see a statistical summary of data
print(dataframe.groupby("class").size()) # to see class distribution

# DATASET VISUALIZATION

print("STEP 4: DATASET VISUALIZATION")
# Box-and-whisker Plot
dataframe.plot(kind = "box", subplots = True, layout = (2, 2), sharex = False,\
               sharey = False)
pyplot.show() 
# Histogram
dataframe.hist()
pyplot.show()
# Scatter Plot Matrix
scatter_matrix(dataframe)
pyplot.show()

# ALGORITHM SELECTION

print("STEP 5: ALGORITHM SELECTION")
## Splitting Input/Output Data
array = dataframe.values
X = array[:, 0:4]
Y = array[:, 4]
## Splitting Train/Test Data
test_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = test_size, random_state = seed)
# Algorithms to Evaluate
models = []
models.append(("CART", DecisionTreeClassifier()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("LR", LogisticRegression()))
models.append(("NB", GaussianNB()))
# Evaluate Each Algorithm Numerically
results = []
names = []
for name, model in models: 
    kfold = KFold(n_splits=10, random_state=seed) 
    cv_result = cross_val_score(model, X_train, Y_train, cv=kfold, \
                                scoring='accuracy') 
    results.append(cv_result) 
    names.append(name) 
    msg = "%s: %f +/-%f" % (name, cv_result.mean(), cv_result.std()) 
    print(msg)
# Evaluate Each Algorithm Graphically
figr = pyplot.figure()
figr.suptitle("Evaluate ALgorithms")
axes = figr.add_subplot(111)
pyplot.boxplot(results)
axes.set_xticklabels(names)
pyplot.show()

# ALGORITHM PREDICTION

print("STEP 6: ALGORITHM PREDICTION")
knnc = KNeighborsClassifier()
knnc.fit(X_train, Y_train)
prediction = knnc.predict(X_test)
print(accuracy_score(Y_test, prediction))
print(confusion_matrix(Y_test, prediction))
print(classification_report(Y_test, prediction))