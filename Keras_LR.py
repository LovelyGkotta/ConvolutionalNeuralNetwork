# Keras LR
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential #
from keras.layers import Dense
model = Sequential()

X_data = np.random.rand(200)*10
noise = np.random.normal(loc=0,scale=1.3,size=X_data.shape) #中心，离散范围，

Y_data = 2*X_data + noise

X_train, Y_train = X_data[:160], Y_data[:160]     # 前160组数据为训练数据集
X_test, Y_test = X_data[160:], Y_data[160:]       # 后40组数据为测试数据集

model.add(Dense(units=1,input_dim=1))
model.compile(optimizer='sgd',loss='mse')

for step in range(2000):
    train_cost = model.train_on_batch(X_train,Y_train) #train
    if step % 100 == 0:
        print('train_cost:',train_cost)
w,b = model.layers[0].get_weights()
Y_pred = model.predict(X_train)
test_cost = model.evaluate(X_test,Y_test,batch_size=40)

print("test_cost",test_cost)
print('w:',w,'b:',b)
plt.scatter(X_data,Y_data,marker='x')
plt.plot(X_train,Y_pred,c='r')
plt.xlabel('X_data')
plt.ylabel('Y_data')
plt.show()
