import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./data/cansim-0800020-eng-6674700030567901031.csv',
                 skiprows=6, skipfooter=9, engine='python')
#print(df.head()) #불러들인 데이터 확인용
df['Adjustments'] = pd.to_datetime(df['Adjustments']) + MonthEnd(1)
df = df.set_index('Adjustments')
#print(df.head())
#plt.plot(df)
#plt.show()
split_date = pd.Timestamp('01-01-2011') # 시간단위로 쪼개기
train = df.loc[:split_date, ['Unadjusted']]
test = df.loc[split_date:, ['Unadjusted']]
ax = train.plot()
test.plot(ax=ax)
#plt.legend(['train', 'test'])
#plt.plot()
#plt.show()
sc = MinMaxScaler()
train_sc = sc.fit_transform(train) # 주가를 0과 1 사이의 무한대 실수값으로 변형
test_sc = sc.transform(test)
train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)
#print(train_sc_df.head()) #스케일링 결과 확인

# pandas shift를 통해 window만들기
# shift 는 이전 정보를 다음 row 에서 다시 쓰기 위한 pandas 함수
# 과거의 값을 총 12개로 저장하며, timestep은 12개가 됨
# 이 작업의 이유는 shift 1 ~ 12 를 통해 현재값 scaled 를 예측하는 것
for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['Scaled'].shift(s)
#print(train_sc_df.head(13))
X_train = train_sc_df.dropna().drop('Scaled', axis=1)
y_train = train_sc_df.dropna()[['Scaled']]
X_test = test_sc_df.dropna().drop('Scaled', axis=1)
y_test = test_sc_df.dropna()[['Scaled']]
#최종 트레이닝 셋
print(X_train.head())
print(y_train.head())
# nparray로 변환.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
print(X_train.shape)
print(X_train)
print(y_train.shape)
print(y_train)
print('**********')
X_train_t = X_train.reshape(X_train.shape[0], 12,1)
X_test_t = X_test.reshape(X_test.shape[0], 12,1)
print('최종 Data')
print(X_train.shape)
print(X_train)
print(y_train.shape)
# LSTM 모델 만들기
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping

K.clear_session()
model = Sequential
model.add(LSTM(20, input_shape=(12, 1))) # (timestamp, feature) 12개의 시간간격, 특징은 가격만 활용
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary()) #모델이 잘 생성되었는지 모델요약으로 확인

