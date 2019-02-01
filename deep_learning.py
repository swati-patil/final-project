import numpy as np
import pandas as pd

reviews = pd.read_csv('final_data.csv', error_bad_lines=False, index_col=False, dtype='unicode')
print(reviews.head())

reviews = reviews.drop("categories", axis=1)
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
reviews = reviews.drop("asin", axis=1)
reviews = reviews.drop("price", axis=1)

X = reviews.drop("title", axis=1)
y = reviews_1["price"]
print(X.shape, y.shape)
X = X.dropna()
print(X.dtypes)

#X['index'] = X['index'].astype(float, raise_on_error=False)
#X['overall'] = X['overall'].astype(float, raise_on_error=False)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)

#X_scaler = StandardScaler().fit(X_train)
#X_train_scaled = X_scaler.transform(X_train)
#X_test_scaled = X_scaler.transform(X_test)


# Step 1: Label-encode data set
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
encoded_y_train = label_encoder.transform(y_train)
print(y_train)
y_test = y_test.map(lambda s: '44.99' if s not in label_encoder.classes_ else s)
encoded_y_test = label_encoder.transform(y_test)

# Step 2: Convert encoded labels to one-hot-encoding
y_train_categorical = to_categorical(encoded_y_train)
y_test_categorical = to_categorical(encoded_y_test)

# deep learning model
from keras.models import Sequential
from keras.layers import Dense

# Create model and add layers
model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=4))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=y_train_categorical.shape[1], activation='softmax'))


print("Column names")

print(X_train.columns)

print('-----------------------')
print(y_train_categorical)
print('-------------------------')

X_train = X_train.dropna()

#X_train['overall'] = X_train['overall'].astype(float)
#X_train['cat_encode'] = X_train['cat_encode'].astype(float)

# Compile and fit the mode
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(
    X_train,
    y_train_categorical,
    epochs=60,
    shuffle=True,
    verbose=2
)
