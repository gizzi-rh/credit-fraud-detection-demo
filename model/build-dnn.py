from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation


def buildDnn():
    model = Sequential()
    model.add(Dense(32, name='dense', activation = 'relu', input_dim = len(X.columns)))
    model.add(Dropout(0.2))
    model.add(Dense(32, name='dense_02'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, name='dense_03'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, name='dense_04', activation = 'sigmoid'))
    model.compile(optimizer='SGD',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    model.save('sequential-model.keras')


if __name__ == '__main__':
    buildDnn()