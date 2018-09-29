import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

olympics_csv = pd.read_csv('athlete_events.csv')

df = pd.DataFrame(olympics_csv)

df['Medal'] = df.groupby(['Medal']).ngroup()
df['Name'] = df.groupby(['Name']).ngroup()
df['Sex'] = df.groupby(['Sex']).ngroup()
df['Team'] = df.groupby(['Team']).ngroup()
df['NOC'] = df.groupby(['NOC']).ngroup()
df['Games'] = df.groupby(['Games']).ngroup()
df['Season'] = df.groupby(['Season']).ngroup()
df['City'] = df.groupby(['City']).ngroup()
df['Sport'] = df.groupby(['Sport']).ngroup()
df['Event'] = df.groupby(['Event']).ngroup()

df['Weight'] = df['Weight'].fillna(df['Weight'].mean().astype(int))

df['Height'] = df['Height'].fillna(df['Height'].mean().astype(int))
df['Age'] = df['Age'].fillna(df['Age'].mean().astype(int))


X = np.array(df.iloc[:,0:-1])
Y = np.array([[df['Medal']]])

Y = Y.reshape(271116)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state = 5)

#SGD starts here
clf = linear_model.SGDClassifier()
clf.fit(X, Y)

y_pred = clf.predict(X_test)
print("Accuracy score of SDG: ",accuracy_score(y_test,y_pred))

#Naive bayes starts
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy score of Naive Bayes: ",accuracy_score(y_test,y_pred))

#Kernel approximation starts here
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
clf = SGDClassifier()
clf.fit(X_features, Y)

print("Accuracy score of Kernel Approximation: ",clf.score(X_features, Y))
