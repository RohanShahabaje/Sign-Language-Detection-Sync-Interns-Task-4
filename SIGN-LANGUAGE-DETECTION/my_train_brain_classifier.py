# TRAINING MY CLASSIFIER

import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

my_data_dict = pickle.load(open('./data.pickle','rb'))
# THIS TWO HOLDS MY IMPORTANT THING (ALL COORDINATES)

# print(my_data_dict.keys())
# print(my_data_dict)

# CONVERTING IN NUMPY ARRAYS   
data = np.asarray(my_data_dict['data'])
labels = np.asarray(my_data_dict['labels'])

# NOW SPLITTING FRO TRAIN & TEST

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

my_accuracy_score = accuracy_score(y_pred, y_test)
print(my_accuracy_score*100)

f = open('my_model.p','wb') #  To save our whole Brain(Model)  
pickle.dump({'model':model}, f)
f.close


