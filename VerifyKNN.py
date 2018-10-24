from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
X=iris.data
Y=iris.target
from sklearn.cross_validation import train_test_split
X_tr, X_test, Y_tr, Y_test = train_test_split(X, Y, test_size = .5)

neigh = KNeighborsClassifier(n_neighbors=1, weights='distance')
neigh.fit(X_tr, Y_tr)

if __name__ == '__main__':
	label = neigh.predict(X_test)
	count = 0
	for i in range(len(label)):
		if label[i] == Y_test[i]:
			count += 1
	print((count/len(X_tr))*100)