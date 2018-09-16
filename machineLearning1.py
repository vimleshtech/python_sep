import pandas 
import matplotlib.pyplot as plt
from sklearn import model_selection

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


dataset = pandas.read_csv(url, names=['A', 'B', 'C', 'D', 'class'])



print(dataset)
print(dataset.describe())
print(dataset.groupby('class').size())


# box and whisker plots
#dataset.plot(kind='line', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4] # : - all rows , 0:4  - first four columns
Y = array[:,4]

validation_size = 0.20
seed = 7



X_train, X_validation, Y_train, Y_validation =model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


print(models)
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	
	msg =  name,' '+ str(cv_results.mean())+' '+ str( cv_results.std())
        
	print(msg)











