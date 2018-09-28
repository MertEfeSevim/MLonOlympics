import pandas as pd

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



from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


df_withoutMissing = df[['ID','Name','Sex','Height',
                        'Weight','Team','NOC','Games',
                        'Year','Season','City','Sport',
                        'Event','Medal','Age']].dropna()

age_predicted = df[[]]
df_withoutMissing = df_withoutMissing.dropna()

X = df_withoutMissing.iloc[:,:14]
Y = df_withoutMissing.iloc[:,14]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.4, random_state = 10)

model = GaussianNB()
model.fit(X_train, y_train)

test_data = df_withoutMissing.iloc[:,:14]
age_predicted['Age'] = pd.DataFrame(model.predict(test_data))

df['Age'].fillna(age_predicted['Age'],inplace=True)

#///////

X_main = df.iloc[:,:15]
Y_main = df.iloc[:,15]

X_train, X_test, y_train, y_test = train_test_split(X_main,Y_main, test_size=0.2, random_state = 10)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy score of Naive Bayes: ",accuracy_score(y_test,y_pred))