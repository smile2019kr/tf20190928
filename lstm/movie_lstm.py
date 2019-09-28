from keras import layers, models, datasets #템플릿처럼 모델이 만들어져있는 상태. 데이터를 집어넣어서 customizing하는 수준
from keras.preprocessing import sequence

class Data:
    def __init__(self, max_features=20000, maxlen=80):
        (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(
            num_words=max_features)
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        self.x_train, self.y_train = x_train, y_train #지도학습준비 (데이터생성)
        self.x_test, self.y_test = x_test, y_test

class RNN_LSTM(models.Model):
    #RNN -> text, CNN -> image. + LSTM: 타임라인 적용. DB전체의 단어/이미지를 모두 긁어옴. 정확도높아짐.
    def __init__(self, max_features, maxlen):
        x = layers.Input((maxlen, ))
        h = layers.Embedding(max_features, 128)(x) # text를 읽어오는것? text embedding
        h = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(h) #recurrent : 사이클대로 계속 반복
        y = layers.Dense(1, activation='sigmoid')(h) # dense를 여러개 깔아서 중첩시킬수 있음.
        super().__init__(x, y)
        self.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])# 여러층의 dense 셋팅 후 compile로 확정

class Machine:
    def __init__(self, max_features=20000, maxlen=80):
        self.data = Data(max_features, maxlen)
        self.model = RNN_LSTM(max_features, maxlen)

    def run(self, epoch=3, batch_size=32):
        data = self.data
        model = self.model
        print('Training state')
        model.fit(
            data.x_train,
            data.y_train,
            batch_size=batch_size,
            epochs=epoch,
            validation_data=(data.x_test, data.y_test),
            verbose=2)
        loss, acc = model.evaluate(
            data.x_test,
            data.y_test,
            batch_size=batch_size,
            verbose=2)
        print('Test performance: accuracy={0}, loss={1}'.format(acc, loss))

if __name__ == '__main__':
    m = Machine()
    m.run()