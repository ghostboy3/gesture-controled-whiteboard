import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

#remove all the datapoints with less than 42 points
indices_to_remove = [index for index, element in enumerate(data) if len(element) != 42]
indices_to_remove.reverse()
for index in indices_to_remove:
    del data[index]
    del labels[index]


data = np.asarray(data)
labels = np.asarray(labels)

# Training the model
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()