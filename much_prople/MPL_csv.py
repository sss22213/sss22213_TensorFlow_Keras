import numpy as np
import pandas as pd 
#Label convert to one-hot 
from keras.utils import np_utils
#Read mnist set
from keras.datasets import mnist
#Train
x_Train=pd.read_csv('Train.csv')
x_label=pd.read_csv('Train_label.csv');
x_Train1=x_Train.as_matrix()
x_label1=x_label.as_matrix()

#Test
x_Test=pd.read_csv('Test.csv')
x_Test_label=pd.read_csv('Test_label.csv')
x_Test1=x_Test.as_matrix()
x_Test1_label=x_Test_label.as_matrix()

#x_Test=x_Train.reshape(14,7).astype('float32')

x_Train_normalize=x_Train1/255
x_Test_normalize=x_Test1/255

y_Train_OneHot=np_utils.to_categorical(x_label1)
y_Test_OneHot=np_utils.to_categorical(x_Test1_label)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model=Sequential()
model.add(Dense(units=360,input_dim=180,kernel_initializer='normal',activation='relu'))
model.add(Dense(units=360,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=360,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=3,kernel_initializer='normal',activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_Train_normalize,y=y_Train_OneHot,validation_split=0.001,epochs=50,batch_size=200,verbose=2)

import matplotlib.pyplot as plt 
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
score=model.evaluate(x_Test_normalize,y_Test_OneHot)
print()
print('accuracy=',score[1])
model.save('location_module.h5')
print()
prediction=model.predict_classes(x_Test_normalize)
a=x_Test1_label.ravel()
print()
ps=pd.crosstab(a,prediction,rownames=['labels'],colnames=['predict'])
print(ps)
#prediction=model.predict_classes(x_Test1)
#pd.crosstab(y_Test_OneHot,prediction,colnames=['predict'],rownames=['label'])
