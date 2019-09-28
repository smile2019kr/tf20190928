import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

class KerasSample:
    def __init__(self):
        self.batch_size = 128
        self.num_classes = 10
        self.epochs = 20

        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.x_train /= 255 #누적연산. 나누기를 계속함
        self.x_test /= 255
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)

    def create_model(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))
#        print(model.summary())  # 생성된 모델이 어떤 상태인지 확인

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        history = model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            validation_data=(self.x_test, self.y_test)) #과거의 훈련된 기록을 계속 보유하도록 함
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1]) # accruacy 가 높으면 강화(1점), 낮으면 처벌(0점)하도록 설정가능


if __name__ == '__main__':
    ks = KerasSample()
    ks.create_model()


