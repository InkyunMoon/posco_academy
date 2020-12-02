import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
# tf.executing_eagerly()

def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data() # 6만개의 트레이닝셋, 1만개의 테스트 셋으로 구성
    
    train_data = np.expand_dims(train_data, axis=-1) # 차원 하나 추가 [batch_size, height, width, channel]
    test_data = np.expand_dims(test_data, axis=-1)
    
    train_data, test_data = normalize(train_data, test_data)
    
    train_labels = to_categorical(train_labels, 10) # 정답들에 대한 처리, shape을 [N, 10]으로 바꾸기. Y의 값이 몇개인지에 따라 다르게 설정할 것. # (10000,) ->(10000,10)으로 변경
    test_labels = to_categorical(test_labels, 10)
    
    return train_data, train_labels, test_data, test_labels
    
def normalize(train_data, test_data):
    train_data = train_data.astype(np.float32)/255.0
    test_data = test_data.astype(np.float32)/255.0
    return train_data, test_data

# 네트워크 구성
def flatten():
    return tf.keras.layers.Flatten()
def dense(channel, weight_init): # fully connected layer를 사용할 것이기 때문에 케라스 레이어에 덴스륾 만들어준다.
    return tf.keras.layers.Dense(units=channel, use_bias=True, kernel_initializer=weight_init) # unit은 아웃풋으로 나가는 채널 개수 설정
# bias = True일 때, 바이어스 사용

def relu():
    return tf.keras.layers.Activation(tf.keras.activations.relu)

class create_model(tf.keras.Model):# 클래스 타입으로 모델을 생성할 때는, tf.keras.Model을 상속해야 한다.
    def __init__(self, label_dim): #네트워크의 아웃풋 차원을 지정 (일반적으로 모델을 사용하기 위해)
        super(create_model, self).__init__()
        
        weight_init = tf.keras_initializers.RandomNormal() # weight normalization의 경우 Random-normal initializer사용 (평균0, 분산1)
        self.model = tf.keras.Sequential()
    
