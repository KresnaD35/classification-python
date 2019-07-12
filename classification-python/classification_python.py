import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score

#datasets X Axis = [refund, marital status, taxable income]; Y Axis = [cheat] 
#refund No = 0; Yes = 1; 
#marital status Single = 0; Married = 1; Divorced = 2;
#cheat No = 0; Yes = 1;

x_train = np.array([[1, 0, 125000], [0, 1, 100000], 
                    [0, 0, 70000], [1, 1, 120000], 
                    [0, 2, 95000], [0, 1, 60000], 
                    [1, 2, 220000], [0, 0, 85000], 
                    [0, 1, 75000], [0, 0, 90000]])

y_train = np.array([0, 0, 0, 0, 1, 
                    0, 0, 1, 0, 1])

x_tes_tree = np.array([[0, 1, 90000], [1, 1, 110000],
                  [0, 2, 65000], [1, 2, 150000],
                  [1, 0, 80000], [0, 0, 180000],
                  [1, 1, 85000], [0, 2, 125000],
                  [1, 2, 62000], [1, 0, 175000]])

clf = tree.DecisionTreeClassifier()

clf = clf.fit(x_train, y_train)

print(y_train)
prediction_tree = clf.predict(x_tes_tree)

accuration_tree = accuracy_score(y_train, prediction_tree)

print(prediction_tree)

