import numpy as np
import matplotlib.pyplot as plt

def Sigmoid(x):  # シグモイド関数
    return 1 / (1 + np.exp(-x))

def dif_Sigmoid(x):  # シグモイド関数の微分
    fx = Sigmoid(x)
    return fx * (1 - fx)

class my_Neural_Network:
    def __init__(self):  # 重みとバイアスの初期化
        self.w1,self.w2,self.w3,self.w4,self.w5,self.w6= 1,1,1,1,1,1
        self.b1,self.b2,self.b3 = 0,0,0

    def feedforward(self,x):  # 順伝播　モデルの構造
        h1 = Sigmoid(x[0] * self.w1 + x[1] * self.w2 + self.b1)
        h2 = Sigmoid(x[0] * self.w3 + x[1] * self.w4 + self.b2)
        out = Sigmoid(h1 * self.w5 + h2 * self.w6 + self.b3)
        return out

    def train(self,train_data,ture_y):  # 逆伝播
        learn_rate = 0.1
        epochs = 2001
        x_epoch = []
        y_loss = []

        for epoch in range(epochs):
            for x,y_true in zip(train_data,ture_y):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = Sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = Sigmoid(sum_h2)

                sum_out = self.w5 * h1 + self.w6 * h2 + self.b3
                out = Sigmoid(sum_out)
                y_pred = out

                d_L_d_y_pred = -2 * (y_true - y_pred)

                d_y_pred_d_h1 = self.w5 * dif_Sigmoid(sum_out)
                d_y_pred_d_h2 = self.w6 * dif_Sigmoid(sum_out)

                d_y_pred_d_w5 = h1 * dif_Sigmoid(sum_out)
                d_y_pred_d_w6 = h2 * dif_Sigmoid(sum_out)
                d_y_pred_d_b3 = dif_Sigmoid(sum_out)

                d_h1_d_w1 = x[0] * dif_Sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * dif_Sigmoid(sum_h1)
                d_h1_d_b1 = dif_Sigmoid(sum_h1)

                d_h2_d_w3 = x[0] * dif_Sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * dif_Sigmoid(sum_h2)
                d_h2_d_b2 = dif_Sigmoid(sum_h2)

                self.w1 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_w2
                self.w3 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_w4
                self.w5 -= learn_rate * d_L_d_y_pred * d_y_pred_d_w5
                self.w6 -= learn_rate * d_L_d_y_pred * d_y_pred_d_w6

                self.b1 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_b1
                self.b2 -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_b2
                self.b3 -= learn_rate * d_L_d_y_pred * d_y_pred_d_b3

            if epoch % 100 == 0:  # 100回訓練ごとLossをプリントする
                y_preds = np.apply_along_axis(self.feedforward, 1, train_data)
                loss = ((true_y - y_preds) ** 2).mean()
                print("Epoch %d loss: %.3f" % (epoch, loss))
                x_epoch.append(epoch)
                y_loss.append(loss)

            if epoch % 500 == 0:  # 500回ごと重みとバイアスをプリントする
                print("weight w1 %.3f  w2 %.3f  w3 %.3f  w4 %.3f  w5 %.3f w6 %.3f" % (
                self.w1, self.w2, self.w3, self.w4, self.w5, self.w6))
                print("bias b1 %.3f b2 %.3f b3 %.3f" % (self.b1, self.b2, self.b3))
        plt.plot(x_epoch, y_loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")

    def test(train,x):  # 訓練したのニューラルネットワークに身長と体重データを入れて男性と女性を判断する
        h1 = Sigmoid(x[0] * train.w1 + x[1] * train.w2 + train.b1)
        h2 = Sigmoid(x[0] * train.w3 + x[1] * train.w4 + train.b2)
        out = Sigmoid(h1 * train.w5 + h2 * train.w6 + train.b3)
        print("予測結果（１に近くは男性、０に近くは女性）：",out)

train_data = np.array([[5, 11],  # [身長、体重]と平均の偏差、身長体重偏差高いの方が男性、低いの方が女性
                       [12, 8],
                       [22,11],
                       [6, 2],
                       [25, 6],
                       [-7, 1],
                       [-15, -6],
                       [1, 2],
                       [-9,0],
                       [-13,-2]])

true_y = np.array([1,  # １は男性、０は女性
                   1,
                   1,
                   1,
                   1,
                   0,
                   0,
                   0,
                   0,
                   0,])

test = [11,2]

network = my_Neural_Network()
network.train(train_data, true_y)
network.test(test)
plt.show()