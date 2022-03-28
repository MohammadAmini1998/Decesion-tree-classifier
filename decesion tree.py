
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os 
import warnings
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import graphviz

from IPython.display import Image


warnings.filterwarnings('ignore')

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df=pd.read_csv("car_evaluation.csv",header=None,names=col_names)

# here we see if there is a missing data or not ... ! 
print(df.isnull().sum())

X=df[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]
Y=df[['class']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)


encoder = ce.OrdinalEncoder(cols=['buying', 'maint',
                                 'doors', 'persons',
                                  'lug_boot', 'safety'])
X_train=encoder.fit_transform(X_train)
X_test=encoder.transform(X_test)

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=0)
clf_gini.fit(X_train, Y_train)

test_acc=clf_gini.score(X_test,Y_test)

train_acc=clf_gini.score(X_train,Y_train)


# print('test accuracy score with criterion gini index: {0:0.4f}'.format(test_acc))

# print('train accuracy score with criterion gini index: {0:0.4f}'.format(train_acc))



plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini.fit(X_train,Y_train))
plt.show()






# dot_data = tree.export_graphviz(clf_gini, out_file=None, 
#                               feature_names=X_train.columns,  
#                               class_names=Y_train,  
#                               filled=True, rounded=True,  
#                               special_characters=True)

# graph = graphviz.Source(dot_data) 

# Image(graph.create_png())






x=X_train[0:1]
x=encoder.transform(x)
result=clf_gini.predict(x)
print(result)
