# In[1]:


import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt


# In[2]:


train_df = pd.read_csv(r'C:\Users\hp\Desktop\train.csv')
test_df = pd.read_csv( r'C:\Users\hp\Desktop\test.csv' )

#Reading data using Pandas
X_train=  (train_df.iloc[:,1:].values).astype('float32')
print(X_train.shape)
Y_train = (train_df.iloc[:,0].values).astype('int32')
X_test=  (test_df.iloc[:,:].values).astype('float32')
print(X_test.shape)
#Y_test = (test_df.iloc[:,0].values).astype('int32')


# In[3]:


from keras.utils.np_utils import to_categorical
X_train = X_train/255
X_test = X_test/255
Y_train = to_categorical(Y_train)


# In[5]:


from keras.models import Sequential
from keras.layers import Dense,Activation
model = Sequential()
# Note here that input_shape does not take number of training samples as input
# Adding Activation as a separate layer is same as specifing the activation parameter in Dense layer

model.add(Dense(5, activation = 'relu', input_shape= X_train.shape[1:]))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(10, activation='softmax'))


# In[6]:


from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001),
loss='categorical_crossentropy',
metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=10, verbose=1)


# In[7]:


print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['acc'])
plt.title('model loss')
plt.ylabel('loss/acc')
plt.xlabel('epoch')
plt.legend(['loss', 'acc'])
plt.show()


# In[8]:


predictions = model.predict(X_test)
predicted_vals = np.argmax(predictions, axis = 1)


# In[9]:


img = X_test[5]
img = np.reshape(img, (28, 28) )
plt.imshow(img)
print("Predicted vals: " + str(predicted_vals[5]))


# In[10]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

seed = 7
np.random.seed(seed)

# note that  tensorflow backend accepts input in form of (rows, columns, channels) wheres theano accepts it as (chnanel, rows, columns)
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=( 28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[11]:


# build the model
model = create_model()
X_train =  np.reshape(X_train, ( X_train.shape[0], 28,  28, 1) )
# Fit the model
history = model.fit(X_train, Y_train, epochs=10, validation_split= 0.3, batch_size=200, verbose=1)
X_test =  np.reshape(X_test, ( X_test.shape[0], 28,  28, 1) )
print(X_test.shape)


# In[12]:


predictions = model.predict(X_test)
predicted_vals = np.argmax(predictions, axis = 1)
predicted_vals = predicted_vals


# In[16]:


print(predicted_vals)
results = pd.Series(predicted_vals,name="Label")
submission = pd.concat([pd.Series(range(1,len(predicted_vals) + 1 ),name = "ImageId"),results],axis =1 )
submission.to_csv(r"C:\Users\hp\Desktop\cnn_mnist_predict_new.csv",index=False)





