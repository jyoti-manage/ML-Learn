import imp
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

print(iris.DESCR)

features = iris.data
labels = iris.target

clf = KNeighborsClassifier()
clf.fit(features,labels)

preds = clf.predict([[1,1,1,1]])
print(preds)