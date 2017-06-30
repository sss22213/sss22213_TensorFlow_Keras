import numpy as np
import pandas as pd 
#Label convert to one-hot 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
model = load_model('location_module.h5')

#Test
x_Test=pd.read_csv('Trail.csv')
x_Test_label=pd.read_csv('Trail_label.csv')
x_Test1=x_Test.as_matrix()
x_Test1_label=x_Test_label.as_matrix()

x_Test_normalize=x_Test1/255

y_Test_OneHot=np_utils.to_categorical(x_Test1_label)

score=model.evaluate(x_Test_normalize,y_Test_OneHot)
print()
print('accuracy=',score[1])
prediction=model.predict_classes(x_Test_normalize)
a=x_Test1_label.ravel()
print()
ps=pd.crosstab(prediction,a,rownames=['predict'],colnames=['labels'])
print()
print(ps)