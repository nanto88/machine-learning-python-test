from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Algorithm Model
clf = tree.DecisionTreeClassifier()
clf2 = GaussianNB()

# Training data
clf = clf.fit(X,Y)
clf2 = clf2.fit(X,Y)

# Prediction
prediction = clf.predict([[170, 70, 44]])
prediction2 = clf2.predict([[170, 70, 44]])

# Accuracy Score
#accuracy = accuracy_score(prediction2, X)
#accuracy2 = accuracy_score(prediction2, X)
print(prediction2, prediction)
