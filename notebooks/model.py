data_sorted = [] #Загрузка из модуля обработки

import numpy as np
import pandas as pd

# Определим функцию для создания последовательностей
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix, :-1], data[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Определите размер шага для последовательностей
n_steps = 10

# Инициализируйте списки для сбора последовательностей и меток
X, y = [], []

# Проходим по каждому уникальному контрагенту и создаем последовательности
for контрагент in data_sorted['ИД_Контрагент'].unique():
    контрагент_data = data_sorted[data_sorted['ИД_Контрагент'] == контрагент]
    # Преобразуем данные контрагента в массив numpy для обработки
    контрагент_values = контрагент_data.drop(['ИД_Контрагент'], axis=1).values
    # Создаем последовательности для текущего контрагента
    for i in range(len(контрагент_values)):
        end_ix = i + n_steps
        if end_ix > len(контрагент_values):
            break
        seq_x, seq_y = контрагент_values[i:end_ix, :-1], контрагент_values[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)

# Преобразуйте списки в массивы NumPy для обучения модели
X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split

# Разделим данные на две группы: те, кто прекратил заказывать, и те, кто продолжает заказывать
stopped_ordering = data_sorted[data_sorted['CustomerStoppedOrdering'] == 1]
continuing_ordering = data_sorted[data_sorted['CustomerStoppedOrdering'] == 0]

# Разделим данные каждой группы на обучающий, валидационный и тестовый наборы данных
train_stopped, test_stopped = train_test_split(stopped_ordering, test_size=0.3, random_state=42)
train_continue, test_continue = train_test_split(continuing_ordering, test_size=0.3, random_state=42)

# Разделим обучающие данные для каждой группы на обучающий и валидационный наборы данных
train_stopped, val_stopped = train_test_split(train_stopped, test_size=0.2, random_state=42)
train_continue, val_continue = train_test_split(train_continue, test_size=0.2, random_state=42)

# Объединим данные для каждого набора вместе
X_train = pd.concat([train_stopped, train_continue])
X_val = pd.concat([val_stopped, val_continue])
X_test = pd.concat([test_stopped, test_continue])

# Определим соответствующие метки для каждого набора данных
y_train = X_train['CustomerStoppedOrdering'].values
y_val = X_val['CustomerStoppedOrdering'].values
y_test = X_test['CustomerStoppedOrdering'].values

# Удалим метки из наборов данных
X_train = X_train.drop(columns=['CustomerStoppedOrdering'])
X_val = X_val.drop(columns=['CustomerStoppedOrdering'])
X_test = X_test.drop(columns=['CustomerStoppedOrdering'])

# Добавляем дополнительное измерение для создания трехмерного массива
X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Определяем размерности входных данных
n_steps = X_train.shape[1]  # количество временных шагов в каждой последовательности
n_features = X_train.shape[2]  # количество признаков в данных


# Создаем DataFrame из массивов NumPy
X_train_df = pd.DataFrame(X_train.reshape(X_train.shape[0], X_train.shape[1]))
X_val_df = pd.DataFrame(X_val.reshape(X_val.shape[0], X_val.shape[1]))
X_test_df = pd.DataFrame(X_test.reshape(X_test.shape[0], X_test.shape[1]))

# Преобразуем все значения в числовой формат
X_train_df = X_train_df.apply(pd.to_numeric)
X_val_df = X_val_df.apply(pd.to_numeric)
X_test_df = X_test_df.apply(pd.to_numeric)

# Преобразуем обратно в массивы NumPy
X_train = X_train_df.values
X_val = X_val_df.values
X_test = X_test_df.values

# Изменяем размерность X_train
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_reshaped = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

from keras.models import Sequential
from keras.layers import LSTM, Dense

# Определим размерности входных данных
n_features = X_train_reshaped.shape[2]  # количество признаков в данных
n_steps = X_train_reshaped.shape[1]  # количество временных шагов в каждой последовательности

# Определим архитектуру модели LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1, activation='sigmoid'))

# Компилируем модель
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучаем модель
history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_data=(X_val_reshaped, y_val))

X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Оценим модель на тестовых данных
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print("Точность модели на тестовых данных:", accuracy)

from keras.models import load_model

# Сохранение модели
model.save('Sales_model.h5')