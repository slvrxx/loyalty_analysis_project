import pandas as pd

# Загрузка данных
df1 = pd.read_csv('/Users/igortarasov/Downloads/loyalty_analysis_project/data/source/SaleData_p1.txt', delimiter='\t')
df2 = pd.read_csv('/Users/igortarasov/Downloads/loyalty_analysis_project/data/source/SaleData_p2.txt', delimiter='\t')

# Объединение двух DataFrame в один
combined_df = pd.concat([df1, df2], ignore_index=True)

# Проверим наличие пустых значений
missing_values = combined_df.isnull().sum()
missing_values

# Работа с пропусками
# Удаляем строки, где 'ИД_Грузополучатель' отсутствует
df_cleaned = combined_df.dropna(subset=['ИД_Грузополучатель'])

# Заменяем пропущенные значения в 'ТипНоменклатуры' на 'Прочее'
df_cleaned['ТипНоменклатуры'].fillna('Прочее', inplace=True)

# Удаляем строки, где 'Количество' или 'Стоимость' пропущены
df_cleaned = df_cleaned.dropna(subset=['Количество', 'Стоимость'])

# Удаляем строки, где отсутствует 'ЗаказПокупателяНомер' или 'ЗаказПокупателяДата'
df_cleaned = df_cleaned.dropna(subset=['ЗаказПокупателяНомер', 'ЗаказПокупателяДата'])

# Проверяем еще раз наличие пустых значений после очистки
missing_values_after_cleaning = df_cleaned.isnull().sum()
df_cleaned.head(), missing_values_after_cleaning

# Выведем уникальные значения столбца 'ПризнакКонтрагента'
unique_values = df_cleaned['ПризнакКонтрагента'].unique()
unique_values

# Заменяем значения 'сети' на 0 в столбце 'ПризнакКонтрагента'
df_cleaned['ПризнакКонтрагента'].replace('сети', 0, inplace=True)

# Выведем уникальные значения после замены для проверки
updated_values = df_cleaned['ПризнакКонтрагента'].unique()
updated_values

# Переименовываем столбцы
df_cleaned.rename(columns={'ЗаказПокупателяДата': 'ДатаЗаказа', 'Период': 'ДатаОтгрузки'}, inplace=True)

# Конвертируем строки в даты
df_cleaned['ДатаЗаказа'] = pd.to_datetime(df_cleaned['ДатаЗаказа'], dayfirst=True).dt.strftime('%d%m%Y')
df_cleaned['ДатаОтгрузки'] = pd.to_datetime(df_cleaned['ДатаОтгрузки'], dayfirst=True).dt.strftime('%d%m%Y')

# Проверяем результаты
df_cleaned[['ДатаЗаказа', 'ДатаОтгрузки']].head()

# Конвертируем строки в даты с правильным форматом
df_cleaned['ДатаЗаказа'] = pd.to_datetime(df_cleaned['ДатаЗаказа'], format='%d%m%Y').dt.strftime('%d.%m.%Y')
df_cleaned['ДатаОтгрузки'] = pd.to_datetime(df_cleaned['ДатаОтгрузки'], format='%d%m%Y').dt.strftime('%d.%m.%Y')

# Проверяем результаты
df_cleaned[['ДатаЗаказа', 'ДатаОтгрузки']].head()

# Проверяем типы данных в столбце 'ДатаЗаказа' и если требуется, преобразуем их обратно в datetime
df_cleaned['ДатаЗаказа'] = pd.to_datetime(df_cleaned['ДатаЗаказа'], format='%d.%m.%Y')

# Сортировка необходима для корректного расчета интервалов
df_cleaned = df_cleaned.sort_values(by=['ИД_Контрагент', 'ДатаЗаказа'])

# Расчет интервалов между заказами для каждого контрагента и грузополучателя
df_cleaned['Разница_дней_контрагент'] = df_cleaned.groupby('ИД_Контрагент')['ДатаЗаказа'].diff().dt.days
df_cleaned['Разница_дней_грузополучатель'] = df_cleaned.groupby('ИД_Грузополучатель')['ДатаЗаказа'].diff().dt.days

# Преобразование интервалов в месяцы без заказов (более 30 дней считается как 1 месяц)
df_cleaned['Месяцы_без_заказов_контрагент'] = df_cleaned['Разница_дней_контрагент'].apply(lambda x: 1 if x >= 30 else 0)
df_cleaned['Месяцы_без_заказов_грузополучатель'] = df_cleaned['Разница_дней_грузополучатель'].apply(lambda x: 1 if x >= 30 else 0)

# Агрегация данных для получения общего количества месяцев без заказов для каждого контрагента и грузополучателя
агрегированные_данные_контрагент = df_cleaned.groupby('ИД_Контрагент')['Месяцы_без_заказов_контрагент'].sum().reset_index()
агрегированные_данные_грузополучатель = df_cleaned.groupby('ИД_Грузополучатель')['Месяцы_без_заказов_грузополучатель'].sum().reset_index()

# Смотрим на агрегированные результаты
(агрегированные_данные_контрагент.head(), агрегированные_данные_грузополучатель.head())

# Присоединяем агрегированные данные к основному датафрейму
df1_enriched = df_cleaned.merge(агрегированные_данные_контрагент, on='ИД_Контрагент', how='left')
df1_enriched = df1_enriched.merge(агрегированные_данные_грузополучатель, on='ИД_Грузополучатель', how='left')

# Удаляем указанные столбцы из обогащенного датафрейма
df1_enriched.drop(['Разница_дней_контрагент', 'Разница_дней_грузополучатель', 'Месяцы_без_заказов_контрагент_x', 'Месяцы_без_заказов_грузополучатель_x'], axis=1, inplace=True)

# Переименовываем столбцы
df1_enriched.rename(columns={'Месяцы_без_заказов_контрагент_y': 'SumMonthWithOutOrderCustomer',
                             'Месяцы_без_заказов_грузополучатель_y': 'SumMonthWithOutOrdeCargo'}, inplace=True)

import numpy as np

# Определяем максимальную дату в датафрейме
max_date = df1_enriched['ДатаЗаказа'].max()

# Добавляем признак для контрагента
max_date_per_customer = df1_enriched.groupby('ИД_Контрагент')['ДатаЗаказа'].max()
df1_enriched['CustomerStoppedOrdering'] = df1_enriched['ИД_Контрагент'].apply(lambda x: 1 if (max_date - max_date_per_customer[x]).days > 90 else 0)

# Добавляем признак для грузополучателя
max_date_per_cargo = df1_enriched.groupby('ИД_Грузополучатель')['ДатаЗаказа'].max()
df1_enriched['CargoStoppedOrdering'] = df1_enriched['ИД_Грузополучатель'].apply(lambda x: 1 if (max_date - max_date_per_cargo[x]).days > 90 else 0)

# Проверяем результаты
df1_enriched[['ИД_Контрагент', 'CustomerStoppedOrdering', 'ИД_Грузополучатель', 'CargoStoppedOrdering']].head()

# Сохраняем обогащенный датафрейм в формате CSV
df1_enriched.to_csv('/Users/igortarasov/Downloads/loyalty_analysis_project/data/enriched_data.csv', index=False)