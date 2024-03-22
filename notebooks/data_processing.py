import pandas as pd

# Загрузка данных из файла
data_path = '/Users/igortarasov/Downloads/loyalty_analysis_project/data/enriched_data.csv'
data = pd.read_csv(data_path)

# Отображение первых строк данных для оценки структуры и потенциальных проблем
data.head(), data.info(), data.describe()

# Коррекция преобразования для учета чисел с плавающей точкой
data['Количество'] = data['Количество'].apply(lambda x: float(x.replace('\xa0', '').replace(' ', '').replace(',', '.')))
data['Стоимость'] = data['Стоимость'].apply(lambda x: float(x.replace('\xa0', '').replace(' ', '').replace(',', '.')))

# Повторная проверка на пропуски после преобразования
missing_values_corrected = data.isnull().sum()

# Повторное отображение результатов после коррекции
data.head(), missing_values_corrected

import matplotlib.pyplot as plt
import seaborn as sns

# Установка стиля для графиков
sns.set(style="whitegrid")

# Создание фигуры для визуализации
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Распределение ПризнакКонтрагента
sns.countplot(data=data, x='ПризнакКонтрагента', ax=axes[0, 0])
axes[0, 0].set_title('Распределение ПризнакКонтрагента')

# Распределение CustomerStoppedOrdering
sns.countplot(data=data, x='CustomerStoppedOrdering', ax=axes[0, 1])
axes[0, 1].set_title('Распределение CustomerStoppedOrdering')

# Распределение CargoStoppedOrdering
sns.countplot(data=data, x='CargoStoppedOrdering', ax=axes[1, 0])
axes[1, 0].set_title('Распределение CargoStoppedOrdering')

# Распределение SumMonthWithOutOrderCustomer
sns.histplot(data=data, x='SumMonthWithOutOrderCustomer', bins=20, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Распределение SumMonthWithOutOrderCustomer')

plt.tight_layout()
plt.show()

# Сортировка данных по контрагенту, грузополучателю и дате заказа
data_sorted = data.sort_values(by=['ИД_Контрагент', 'ИД_Грузополучатель', 'ДатаЗаказа'])

# Расчет динамических признаков
data_sorted['Предыдущее_Количество'] = data_sorted.groupby(['ИД_Контрагент', 'ИД_Грузополучатель'])['Количество'].shift(1)
data_sorted['Изменение_Количества'] = data_sorted['Количество'] - data_sorted['Предыдущее_Количество']

data_sorted['Предыдущая_Стоимость'] = data_sorted.groupby(['ИД_Контрагент', 'ИД_Грузополучатель'])['Стоимость'].shift(1)
data_sorted['Изменение_Стоимости'] = data_sorted['Стоимость'] - data_sorted['Предыдущая_Стоимость']

# Заполнение пропусков, возникающих из-за отсутствия предыдущего значения для первого заказа
data_sorted['Изменение_Количества'] = data_sorted['Изменение_Количества'].fillna(0)
data_sorted['Изменение_Стоимости'] = data_sorted['Изменение_Стоимости'].fillna(0)

# Отображение результатов с новыми признаками
data_sorted[['ИД_Контрагент', 'ИД_Грузополучатель', 'Количество', 'Предыдущее_Количество', 'Изменение_Количества', 'Стоимость', 'Предыдущая_Стоимость', 'Изменение_Стоимости']].head()


# Добавление ТипНоменклатуры в группировку
data_sorted['Предыдущее_Количество'] = data_sorted.groupby(['ИД_Контрагент', 'ИД_Грузополучатель', 'ТипНоменклатуры'])['Количество'].shift(1)
data_sorted['Изменение_Количества'] = data_sorted['Количество'] - data_sorted['Предыдущее_Количество']

data_sorted['Предыдущая_Стоимость'] = data_sorted.groupby(['ИД_Контрагент', 'ИД_Грузополучатель', 'ТипНоменклатуры'])['Стоимость'].shift(1)
data_sorted['Изменение_Стоимости'] = data_sorted['Стоимость'] - data_sorted['Предыдущая_Стоимость']

# Заполнение пропусков
data_sorted['Изменение_Количества'] = data_sorted['Изменение_Количества'].fillna(0)
data_sorted['Изменение_Стоимости'] = data_sorted['Изменение_Стоимости'].fillna(0)

# Проверка результатов с учетом типа номенклатуры
data_sorted[['ИД_Контрагент', 'ИД_Грузополучатель', 'ТипНоменклатуры', 'Количество', 'Предыдущее_Количество', 'Изменение_Количества', 'Стоимость', 'Предыдущая_Стоимость', 'Изменение_Стоимости']].head()

# Повторная генерация столбца 'Предыдущая_ДатаЗаказа'
data_sorted['Предыдущая_ДатаЗаказа'] = data_sorted.groupby('ИД_Грузополучатель')['ДатаЗаказа'].shift(1)

# Проверка наличия и корректность данных в 'Предыдущая_ДатаЗаказа'
data_sorted[['ИД_Грузополучатель', 'ДатаЗаказа', 'Предыдущая_ДатаЗаказа']].head()

# Убедимся, что даты в правильном формате
data_sorted['ДатаЗаказа'] = pd.to_datetime(data_sorted['ДатаЗаказа'])
data_sorted['Предыдущая_ДатаЗаказа'] = pd.to_datetime(data_sorted['Предыдущая_ДатаЗаказа'])

# Расчет количества дней между заказами
data_sorted['Дни_Между_Заказами'] = (data_sorted['ДатаЗаказа'] - data_sorted['Предыдущая_ДатаЗаказа']).dt.days

# Заполнение пропусков нулями
data_sorted['Дни_Между_Заказами'] = data_sorted['Дни_Между_Заказами'].fillna(0)

# Проверка результатов добавления нового признака
data_sorted[['ИД_Грузополучатель', 'ДатаЗаказа', 'Предыдущая_ДатаЗаказа', 'Дни_Между_Заказами']].head()

# Расчет корреляции между новыми признаками и целевыми переменными
correlation_matrix = data_sorted[['Изменение_Количества', 'Изменение_Стоимости', 'Дни_Между_Заказами', 'CustomerStoppedOrdering', 'CargoStoppedOrdering']].corr()

# Визуализация корреляционной матрицы
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Корреляционная матрица для новых признаков и целевых переменных')
plt.show()

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Создаем объект LabelEncoder для каждого категориального признака
label_encoders = {}
for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    data_sorted[feature] = label_encoders[feature].fit_transform(data_sorted[feature])

# Нормализация числовых признаков
scaler = MinMaxScaler()
data_sorted[numeric_features] = scaler.fit_transform(data_sorted[numeric_features])

# Преобразуем даты в числовой формат
data_sorted['ДатаЗаказа'] = (data_sorted['ДатаЗаказа'] - data_sorted['ДатаЗаказа'].min()).dt.days
data_sorted['ДатаОтгрузки'] = (pd.to_datetime(data_sorted['ДатаОтгрузки']) - pd.to_datetime(data_sorted['ДатаОтгрузки']).min()).dt.days

# Преобразуем даты в числовой формат
data_sorted['Предыдущая_ДатаЗаказа'] = (data_sorted['Предыдущая_ДатаЗаказа'] - data_sorted['Предыдущая_ДатаЗаказа'].min()).dt.days

# Заменяем пропущенные значения на 0
data_sorted.fillna(0, inplace=True)

# Проверяем, остались ли еще пропущенные значения
missing_values_after = data_sorted.isnull().sum()
print("Пропущенные значения после замены на 0:\n", missing_values_after)

# Отсортируем данные по 'ИД_Контрагент' и 'ДатаЗаказа'
data_sorted = data_sorted.sort_values(by=['ИД_Контрагент', 'ДатаЗаказа'])