import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

reviews = pd.read_csv('msm.csv', error_bad_lines=False, index_col=False, dtype='unicode')
print(reviews.head())

reviews = reviews.dropna()


reviews['cat_encode'] = 0
trans = {'No Title': 0, ' Travel': 1, ' Sports & Outdoors': 2, ' Health': 3, ' Computers & Technology': 4, ' Politics & Social Sciences': 5, ' Education & Teaching': 6, ' New': 7, ' Childrens Books': 8, ' Biographies & Memoirs': 9, ' Literature & Fiction': 10, ' Crafts': 11, ' Cookbooks': 12, ' Science Fiction & Fantasy': 13, ' Reference': 14, ' Arts & Photography': 15, ' Humor & Entertainment': 16, ' Medical Books': 17, ' History': 18, ' Science & Math': 19, ' Business & Money': 20, ' Teen & Young Adult': 21, ' Comics & Graphic Novels': 22, ' Engineering & Transportation': 23, ' Calendars': 24, ' Self-Help': 25, ' Christian Books & Bibles': 26, ' Parenting & Relationships': 27, ' Religion & Spirituality': 28, ' Law': 29, ' Mystery': 30, ' Gay & Lesbian': 31, ' Romance': 32, ' Crafts & Sewing': 33, ' Exterior Accessories': 34}
for i in reviews.index:
    cat = reviews.at[i, 'final_cat']
    print(cat)
    print(i)  
    reviews.at[i, 'cat_encode'] = trans[cat]
#print(reviews.columns)
 
X = reviews[["price", "cat_encode"]] 
y = reviews["overall"]

rf = RandomForestClassifier(n_estimators=500)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)

rf = rf.fit(X_test, y_test)
print('-----------------------------------------------')
print(rf.score(X_train, y_train))
print('-----------------------------------------------')

importances = rf.feature_importances_
print("--------------------------------------------")
print(importances)
print("--------------------------------------------")

filename = 'random-forest-model.sav'
pickle.dump(rf, open(filename, 'wb'))

#test random forest model with custom values
loaded_model = pickle.load(open(filename, 'rb'))
x_t = [[14.99, 10]]

result = loaded_model.predict(x_t)
print(result)