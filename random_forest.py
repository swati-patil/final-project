import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

reviews = pd.read_csv('final_data.csv', error_bad_lines=False, index_col=False, dtype='unicode')
print(reviews.head())

#reviews = reviews.drop("categories", axis=1)
#reviews = reviews.fillna(value="No Title")
reviews = reviews.dropna()


reviews['cat_encode'] = 0
trans = {'No Title': 0, ' Travel': 1, ' Sports & Outdoors': 2, ' Health': 3, ' Computers & Technology': 4, ' Politics & Social Sciences': 5, ' Education & Teaching': 6, ' New': 7, ' Childrens Books': 8, ' Biographies & Memoirs': 9, ' Literature & Fiction': 10, ' Crafts': 11, ' Cookbooks': 12, ' Science Fiction & Fantasy': 13, ' Reference': 14, ' Arts & Photography': 15, ' Humor & Entertainment': 16, ' Medical Books': 17, ' History': 18, ' Science & Math': 19, ' Business & Money': 20, ' Teen & Young Adult': 21, ' Comics & Graphic Novels': 22, ' Engineering & Transportation': 23, ' Calendars': 24, ' Self-Help': 25, ' Christian Books & Bibles': 26, ' Parenting & Relationships': 27, ' Religion & Spirituality': 28, ' Law': 29, ' Mystery': 30, ' Gay & Lesbian': 31, ' Romance': 32, ' Crafts & Sewing': 33, ' Exterior Accessories': 34}
for i in reviews.index:
    cat = reviews.at[i, 'final_cat']
    #print(cat)
    #print(i)
    reviews.at[i, 'cat_encode'] = trans[cat]
print(reviews.columns)

reviews_1 = reviews
reviews = reviews.drop("final_cat", axis=1)
#reviews = reviews.drop("asin", axis=1)
reviews = reviews.drop("price", axis=1)

X = reviews.drop("title", axis=1)
X = X.drop("categories", axis=1)
X = X.drop("asin", axis=1)
        
y = reviews_1["price"]
print(X.columns)

#prices = y["price"].values.tolist()

if 'No Title' in y:
    print("String found in price list")

print("---------------------")

X= X.drop("index", axis=1)
X = X.drop("Unnamed: 0", axis=1)


rf = RandomForestClassifier(n_estimators=200)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)

rf = rf.fit(X_test, y_test)
print('----------------------------------------')

print(rf.score(X_train, y_train))
print('-----------------------------------------------')

importances = rf.feature_importances_
print(importances)
print("--------------------------------------------")

print("--------------------------------------------")

print(X.shape)

print(X.head())

print (y.shape)
print(y.head())

import pickle

filename = 'random-forest-model.sav'
pickle.dump(rf, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))


x_t = [[3.5, 10]]
y_t = [10.00]

result = loaded_model.predict(x_t)

print(result)

print('---------------------------------------------------')
