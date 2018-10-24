from scipy.spatial import distance
from sklearn import datasets

#load test data set and partition into training and test data
iris = datasets.load_iris()
X=iris.data
Y=iris.target
from sklearn.cross_validation import train_test_split
X_tr, X_test, Y_tr, Y_test = train_test_split(X, Y, test_size = .5)



class KNN:

	#Functions for allowing use of max heap to store n closest neighbors
	def max_heapify(self, A, i):
		#takes in a list of tuples, 0 element in tuple is distance
			left = 2*i + 1
			right = 2*i + 2
			largestNode = i 

			#if child is larger than parent, set parent index to child
			if left < len(A) and A[left][0] > A[largestNode][0]:
				largestNode = left
			if right < len(A) and A[right][0] > A[largestNode][0]:
				largestNode = right

			#if child is larger than parent, switch nodes and recurse
			if largestNode != i:
				A[i], A[largestNode] = A[largestNode], A[i]
				self.max_heapify(A, largestNode)

	def make_MaxHeap(self, A):
		'''make Max Heap'''
		for i in range(len(A)//2,-1,-1):
			self.max_heapify(A,i)


	def fit(self, X_train, Y_train):
		'''create variables for training data and associated labels'''
		self.X_train = X_train
		self.Y_train = Y_train


	def closestPt(self, X_feature_set, n):
		'''establish list of closest points. Once full, create max heap
		into which new elements can be inserted in logn time. Iterate over
		all points to find the n closest points'''

		filled = False
		closest_Points = []
		

		for i in range(len(self.X_train)):

			
			Euc_distance = distance.euclidean(X_feature_set, self.X_train[i])
			
			#append to the array if not full
			if len(closest_Points) < n:
				closest_Points.append((Euc_distance, self.Y_train[i]))
			
			#construct max heap
			if len(closest_Points) == n and filled == False:
				self.make_MaxHeap(closest_Points)
				filled = True
			
			#maintain max heap
			elif filled:
				if Euc_distance < closest_Points[0][0]:
					closest_Points[0] = (Euc_distance, self.Y_train[i])
					self.max_heapify(closest_Points, 0)
			
		#return stored closest point label if n = 1
		if len(closest_Points) == 1:
			return closest_Points[0][1]
		
		WeightedDistance = None
		predictedLabel = None
		label_dict = {}

		#iterate over list of closest point
		#uses 1/distance to weight each point
		#stores weight scores in a dictionary
		for item in closest_Points:
			if item[1] not in label_dict.keys():
				if (item[0]) != 0.0:
					label_dict[item[1]] = 1/(item[0])
				elif (item[0]) == 0.0:
					predictedLabel = item[1]
					
					return predictedLabel
			elif item[1] in label_dict.keys():
				if (item[0]) != 0.0:
					label_dict[item[1]] += 1/(item[0])
				elif (item[0]) == 0.0:
					predictedLabel = item[1]
					
					return predictedLabel
			
		#return label with the highest weighted score
		for item in label_dict.keys():
			if WeightedDistance == None:
				WeightedDistance = label_dict[item]
				predictedLabel = item

			if label_dict[item] > WeightedDistance:
				WeightedDistance = label_dict[item]
				predictedLabel = item
		return predictedLabel

	def predict(self, X_feature_set, n=1):
		#find closest point to given X point
		pred_label = self.closestPt(X_feature_set, n)
		return pred_label

	def validateAccuracy(self, X_test, Y_test, rounds=1, neighbors=1):
		#find accuracy of classifier
		for j in range(rounds):
			count = 0
			stored_accuracy = []
			for i in range(len(X_test)):
				label = self.predict(X_test[i],1)
		
				if label == Y_test[i]:
					count +=1
			stored_accuracy.append((count/len(X_test))*100)
		return sum(stored_accuracy)/float(len(stored_accuracy))

	def findOptimalK(self, X_test, Y_test):
		#return optimal value of K for highest accuracy for a dataset
		highest_acccuracy_value = None
		for i in range(1, len(X_test)):
			valAccuracy = self.validateAccuracy(X_test, Y_test, 5, i)
			if highest_acccuracy_value == None:
				highest_acccuracy_value = (i, valAccuracy)
			elif valAccuracy > highest_acccuracy_value[1]:
				highest_acccuracy_value = (i, valAccuracy)
		return highest_acccuracy_value[0]


if __name__ == '__main__':
	
	KNearestN = KNN()
	KNearestN.fit(X_tr, Y_tr)
	print(KNearestN.validateAccuracy(X_test, Y_test, 10, 10))
	print(KNearestN.findOptimalK(X_test, Y_test))

	

			
