import pandas as pd

from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import svm
from sklearn.svm import LinearSVC
#from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import export_graphviz

#read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('dataset.csv')

# "Position (pos)" is the class attribute we are predicting.
class_column = 'Pos'

#The dataset contains attributes such as player name and team name. 
#We know that they are not useful for classification and thus do not 
#include them as features. I selected these attribute values based on
#train and error
feature_columns = ['FGA', '3PA', '2PA', 'FTA', 'ORB', 'DRB', \
    'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']

#Pandas DataFrame allows to select columns. 
#We use column selection to split the data into features and class. 
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)

training_accuracy = []
test_accuracy = []

##            **************KNEIGHBOUR CLASSIFIER**************
#knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', metric='minkowski', p=2)
#knn.fit(train_feature, train_class)
#prediction = knn.predict(test_feature)
#print("Test set predictions:\n{}".format(prediction))
#print("[kNN] Test set accuracy: {:.2f}".format(knn.score(test_feature, test_class)))
#
#prediction = knn.predict(test_feature)
#print("Confusion matrix:")
#print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))
#
#scores = cross_val_score(knn, nba_feature, nba_class, cv=5)
#print("[Linear SVM] Cross-validation scores: {}".format(scores))
#print("[KNN] Average cross-validation score: {:.2f}".format(scores.mean()))

#             **************LINEAR SUPPORT VECTOR MACHINES**************
linearsvm = LinearSVC().fit(train_feature, train_class)
print("[Linear SVM] Test set score: {:.3f}".format(linearsvm.score(test_feature, test_class)))
#Confusion matrix for Linear SVM
prediction = linearsvm.predict(test_feature)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))


##              **************NAIVE BAYES CLASSIFIER**************
#nb = GaussianNB().fit(train_feature, train_class)
#print("[Gaussian NB Classifier] Test set score: {:.3f}".format(nb.score(test_feature, test_class)))
#
#prediction = nb.predict(test_feature)
#print("Confusion matrix:")
#print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))
#
#scores = cross_val_score(nb, nba_feature, nba_class, cv=5)
#print("[Gaussian NB Classifier] Cross-validation scores: {}".format(scores))
#print("[Gaussian NB Classifier] Average cross-validation score: {:.2f}".format(scores.mean()))
#
##             **************DECISION TREE**************
#tree_nba = DecisionTreeClassifier(max_depth=5, min_samples_split=3)
#tree_nba.fit(train_feature, train_class)
#print("[Decision Tree] Training set score: {:.3f}".format(tree_nba.score(train_feature, train_class)))
#print("[Decision Tree] Test set score: {:.3f}".format(tree_nba.score(test_feature, test_class)))
#
#prediction = tree_nba.predict(test_feature)
#print("Confusion matrix:")
#print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))
#
#scores = cross_val_score(tree_nba, nba_feature, nba_class, cv=5)
#print("Cross-validation scores: {}".format(scores))
#print("[Decision Tree] Average cross-validation score: {:.2f}".format(scores.mean()))


#             **************10 FOLD STRATIFIED CROSS VALIDATION :: LINEAR SVM*************
#Cross-validation for 10 fold stratified cross validation
scores = cross_val_score(linearsvm, nba_feature, nba_class, cv=10, scoring='precision_macro')
print("[Linear SVM] Cross-validation scores: {}".format(scores))
print("[Linear SVM] Average cross-validation score: {:.2f}".format(scores.mean()))