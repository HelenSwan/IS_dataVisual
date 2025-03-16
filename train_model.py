import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
Sequential = tf.keras.models.Sequential
load_model = tf.keras.models.load_model
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense

data = pd.read_csv("dataset/most-dangerous-countries-for-women-2024.csv", encoding='ISO-8859-1') 
data2 = pd.read_csv("dataset/Environment_Temperature_change_E_All_Data_NOFLAG.csv", encoding='ISO-8859-1')

# Data cleaning
data.fillna(-1, inplace=True)
data2.fillna(0, inplace=True)

scaler = StandardScaler()
used_data = data[['country', 'MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023',
                    'MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019',
                  'MostDangerousCountriesForWomen_WDIGenderInequality_2019']]
used_data = used_data[(used_data['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023'] != -1) &
                       (used_data['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'] != -1) &
                        (used_data['MostDangerousCountriesForWomen_WDIGenderInequality_2019'] != -1)]

X_scaled = scaler.fit_transform(used_data[['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023',
                                            'MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019',
                                           'MostDangerousCountriesForWomen_WDIGenderInequality_2019']])

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit_predict(X_scaled)
joblib.dump(kmeans, 'kmeans_model.pkl')
used_data['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)
    
# Agglomerative Clustering
agg_clust = AgglomerativeClustering(n_clusters=3)
agg_clust.fit_predict(X_scaled)  
joblib.dump(agg_clust, 'agg_clustering_model.pkl')
# SVM Classifier
X = used_data[['MostDangerousCountriesForWomen_WDIGenderInequality_2019',
           'MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019']]
y = used_data['KMeans_Cluster']
# ฝึกและทดสอบข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# ใช้SVC และเก็บผลการทำนาย
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
joblib.dump(svm, 'svm_model.pkl')
# เลือกเตรียมข้อมูลที่จะใช้
temperature_data_thailand = data2[['Area', 'Y1961', 'Y1964', 'Y1967', 'Y1970', 'Y1973', 'Y1976', 'Y1979', 'Y1982', 'Y1985', 'Y1988', 'Y1991', 'Y1994', 'Y1997', 'Y2000', 'Y2003', 'Y2006', 'Y2009', 'Y2012', 'Y2015', 'Y2018', 'Y2019', 'Element']]
temperature_data_thailand = temperature_data_thailand[temperature_data_thailand['Element'] == 'Temperature change']
temperature_data_thailand = temperature_data_thailand[temperature_data_thailand['Area'] == 'Thailand']  # Filter for Thailand

# แปลงข้อมูลจากปีหลายคอมลัมน์ เป็น 2 คอลัมน์(ปี กับ ค่า Temperature Change)
temperature_data_thailand = temperature_data_thailand.melt(id_vars=['Area', 'Element'], var_name='Year', value_name='TemperatureChange')

# แปลงข้อมูล Y1961 เป็น 1961 ...
temperature_data_thailand['Year'] = temperature_data_thailand['Year'].str.extract('(\d+)').astype(int)
# ใช้แค่คอลัมน์ที่จะใช้ และ ตัดค่าที่ไม่ได้เกี่ยวออก
temperature_data_thailand = temperature_data_thailand[['Area', 'Year', 'TemperatureChange']]
temperature_data_thailand = temperature_data_thailand.dropna()
# จับกลุ่มข้อมูลปีและหาค่าเฉลี่ยของ Temperature Change ในแต่ละปี
temperature_data_thailand_yearly = temperature_data_thailand.groupby('Year').agg({'TemperatureChange': 'mean'}).reset_index()

# เลือกข้อมูลตั้งแต่ปีที่มีเริ่มจนถึงปี2019
data_for_lstm_thailand = temperature_data_thailand_yearly[temperature_data_thailand_yearly['Year'] <= 2019]

# ปรับค่าขนาด Temperature Change
scaler = MinMaxScaler(feature_range=(0, 1))
data_for_lstm_thailand['TemperatureChange'] = scaler.fit_transform(data_for_lstm_thailand['TemperatureChange'].values.reshape(-1, 1))
# เลือกใช้ข้อมูลอุณหภูมิในช่วง 6 ปีที่ผ่านมา
time_step = 6
# ฟังก์ชันเพื่อเตรียมข้อมูลเพื่อ LSTM
def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0]) 
        y.append(dataset[i, 0])  
    return np.array(X), np.array(y)
# ทำ dataset เพื่อ LSTM
dataset = data_for_lstm_thailand['TemperatureChange'].values
dataset = dataset.reshape(-1, 1)  
X, y = create_dataset(dataset, time_step)
# ปรับแบบข้อมูล X ให้เหมาะกับตอนที่จะป้อนข้อมูลเข้า LSTM แล้ว X มีรูปแบบ[samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential()
model.add(LSTM(units=90, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=90, return_sequences=False))
model.add(Dense(units=1)) 

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')
# ฝึก
model.fit(X, y, epochs=10, batch_size=32)
model_path = 'lstm_model.h5'
model.save(model_path)
