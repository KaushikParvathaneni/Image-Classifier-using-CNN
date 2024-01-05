


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split





df=pd.read_pickle('test100c5k_nolabel.pkl')




clf = pd.read_pickle('train100c5k_v2.pkl') 




info =  clf['data']




category = clf['target']





INFO = np.array(clf.data.to_list())
CATEGORY = np.array(clf.target.to_list())






X_train, X_test, y_train, y_test = train_test_split(INFO,CATEGORY, test_size=0.4)




plt.imshow(X_train[12000])




from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPool2D,Dropout
imgclf = Sequential()
imgclf.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
imgclf.add(Conv2D(32, kernel_size=3, activation='relu'))
imgclf.add(Conv2D(128, kernel_size=3, activation='relu'))
imgclf.add(Flatten())
imgclf.add(Dense(100, activation='softmax'))
imgclf.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])




imgclf.summary()




X_train.shape, X_test.shape, y_train.shape, y_test.shape




#imgclf.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)




INFO1 = np.array(df.data.to_list())

imgclf.load_weights("model_kaparvat.h5")



y_pred = imgclf.predict(INFO1)




y_pred



result = np.argmax(y_pred, axis=1)




result




with np.printoptions(threshold=np.inf):
    print(result)




#imgclf.save_weights('model_kaparvat.h5')






