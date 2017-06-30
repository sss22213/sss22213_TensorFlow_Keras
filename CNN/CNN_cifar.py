from keras.datasets import cifar10
import numpy as np 
(x_image_train,y_label_train),(x_image_test,y_label_test)=cifar10.load_data()

load_dict={0:"airplane", 1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

import matplotlib.pyplot as plt
def plot_image_labels_prediction(images,labels,prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title=str(i)+','+load_dict[labels[i][0]]
        if(len(prediction))>0:
            title+='=>'+load_dict[prediction[i]]
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
<<<<<<< HEAD
plot_image_labels_prediction(x_image_train,y_label_train,[],0)
x_image_train_normal=x_image_train.astype('float32')/255.0
x_image_test_normal=x_image_test.astype('float32')/255.0

from keras.utils import np_utils
y_label_train_OneHot=np_utils.to_categorical(y_label_train)
y_label_test_OneHot=np_utils.to_categorical(y_label_test)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Reshape
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(rate=0.25))

model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(10,activation='softmax'))
print (model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_image_train_normal,y=y_label_train_OneHot,validation_split=0.2,epochs=20,batch_size=128,verbose=1)

scores=model.evaluate(x_image_test_normal,y_label_test_OneHot,verbose=0)
print()
print(scores[1])
=======
plot_image_labels_prediction(x_image_train,y_label_train,[],0)
>>>>>>> 176b81030fd03c10db48eb8807162131ea75c091
