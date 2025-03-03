import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

st.title("Machine Learning Models")
st.write(
    "Let's start exploring!"
)
data = pd.read_csv("dataset/most-dangerous-countries-for-women-2024.csv", encoding='ISO-8859-1') 
data2 = pd.read_csv("dataset/Environment_Temperature_change_E_All_Data_NOFLAG.csv", encoding='ISO-8859-1')

# Data cleaning
data.fillna(-1, inplace=True)
data2.fillna(0, inplace=True)
st.header("About Datasets")
st.text("Example The Most Dangerous Countries For Women data in dataset")
st.write(data.head())
st.markdown("""
Download dataset from [Most Dangerous Countries For Women](https://www.kaggle.com/datasets/arpitsinghaiml/most-dangerous-countries-for-women-2024)

### Features of the Dataset:
- **country**: Name of the country
- **MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023**: Women, Peace and Security Index score for 2023. Lower scores indicate greater danger.
- **MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019**: Women's Danger Index total score for 2019. Higher scores indicate greater danger.
- **MostDangerousCountriesForWomen_WDIStreetSafety_2019**: Women's Danger Index score for street safety in 2019.
- **MostDangerousCountriesForWomen_WDIIntentionalHomicide_2019**: Women's Danger Index score for intentional homicide in 2019.
- **MostDangerousCountriesForWomen_WDINonPartnerViolence_2019**: Women's Danger Index score for non-partner violence in 2019.
- **MostDangerousCountriesForWomen_WDIIntimatePartnerViolence_2019**: Women's Danger Index score for intimate partner violence in 2019.
- **MostDangerousCountriesForWomen_WDILegalDiscrimination_2019**: Women's Danger Index score for legal discrimination in 2019.
- **MostDangerousCountriesForWomen_WDIGlobalGenderGap_2019**: Women's Danger Index score for global gender gap in 2019.
- **MostDangerousCountriesForWomen_WDIGenderInequality_2019**: Women's Danger Index score for gender inequality in 2019.
- **MostDangerousCountriesForWomen_WDIAttitudesTowardViolence_2019**: Women's Danger Index score for attitudes toward violence in 2019.
""")
st.text("Example Temperature Change data in dataset")
st.write(data2.head())
st.markdown("""
Download dataset from [Temperature Change Dataset](https://www.kaggle.com/datasets/sevgisarac/temperature-change)

### Features of the Dataset:
- **Area Code**: รหัสพื้นที่
- **Months Code**: รหัสเดือน
- **Months**: ชื่อเดือน
- **Element Code**: รหัสองค์ประกอบ
- **Element**: ประเภทข้อมูล (Temperature change and Standard Deviation)
- **Unit**: หน่วยของข้อมูล (°C)
- **Y1961 - Y2019**: ข้อมูลอุณหภูมิย้อนหลังจากปี 1961 ถึง 2019
""")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["About Machine Learning", "About Neural Network", "Machine Learning Demo", "Neural Network Demo"])
if page  == "About Machine Learning":
#การเตรียมข้อมูล ทฤษฎีของอัลกอริทึมที่พัฒนา และขั้นตอนการพัฒนาโมเดล
    st.title("About Machine Learning")
    st.subheader("Development Plan")
    st.text("ในการทำโมเดลนี้ใช้ 2 Datasets โดย Datasetที่ 1 เกี่ยวกับความเป็นอันตรายและความปลอดภัยของผู้หญิงในประเทศต่าง ๆ และ Datasetที่ 2 เกี่ยวกับการเปลี่ยนแปลงอุณหภูมิของประเทศต่าง ๆ ทั่วโลกปี 1961-2019")
    st.markdown(""" 
**1. Data Preparation**
- เริ่มการอ่านข้อมูลทั้งสองจาก CSV ไฟล์
- ทำความสะอาดข้อมูล Datasetที่ 1 โดยการเติมค่า Missing ในข้อมูลด้วย -1 
- ใน Datasetที่ 1 เลือกใช้เฉพาะคอลัมน์ 'country', 'MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023','MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019','MostDangerousCountriesForWomen_WDIGenderInequality_2019' 
    * ข้อมูลสำหรับ SVM ใช้ X เป็นฟีเจอร์ที่เกี่ยวกับความไม่เท่าเทียมทางเพศและความอันตรายต่อผู้หญิงจากดัชนี(MostDangerousCountriesForWomen_WDIGenderInequality_2019 (ดัชนีความไม่เสมอภาคทางเพศ), MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019 (ดัชนีความอันตรายสำหรับผู้หญิง)) และใช้ y เป็นป้ายชื่อจาก KMeans Clustering ที่แบ่งกลุ่มเป็น 3 กลุ่ม โดยข้อมูลแบ่งออกเป็นชุด training และ testing ใช้ train_test_split
    * แล้วModelทั้งหมดจะคัดข้อมูลที่ไม่มีค่าที่เป็น -1 หลังจากนั้นใช้ StandardScaler เพื่อสเกลข้อมูลให้อยู่ในช่วงที่เหมาะสม ปรับข้อมูลแต่ละตัวแปรให้มีค่าเฉลี่ยเท่ากับ 0 และส่วนเบี่ยงเบนมาตรฐานเท่ากับ 1
                
**2. The Theory of the Developed Algorithm (K-Means Clustering, Agglomerative Clustering, SVM Classifier)**
- **K-Means Clustering**
    * ทฤษฎี-KMeans เป็นอัลกอริธึมการ Clustering ที่ไม่ต้องใช้ Label หรือก็คือ Unsupervised Learning โดยใช้เทคนิคการแบ่งกลุ่มข้อมูลให้เป็น K กลุ่ม ซึ่งอิงจากความคล้ายคลึงกันระหว่างข้อมูลในกลุ่มเดียวกัน ใช้ค่าเฉลี่ยของแต่ละกลุ่ม Centroid เพื่อหาคำนวนความใกล้เคียงระหว่างข้อมูล หรือก็คือพึ่่งระยะห่างระหว่างจุดข้อมูลในการคำนวน
    * ขั้นตอนการพัฒนาโมเดล
        * ทำการเตรียมข้อมูลให้พร้อมกับ KMeans Model ตามที่บอกไว้ในข้างต้น
        * ข้อมูลที่ใช้ใน KMeans คือ Datasetที่ 1 และเลือกจำนวนกลุ่ม K ที่เหมาะสมที่สุดในการทำ KMeans Clustering โดยใช้วิธี Elbow Method ซึ่งเป็นกราฟที่แสดงความสัมพันธ์ระหว่างจำนวนกลุ่มและการกระจายข้อมูล โดยการคำนวณ Within-Cluster Sum of Squares (WCSS) สำหรับจำนวนของ k ตั้งแต่ 1 ถึง 14 เลือกค่า k จากค่าของ WCSS ที่ลดลงอย่างรวดเร็วจากจำนวนค่าสองค่าสุดท้าย
        * เมื่อได้ค่า k เท่ากับ 3 จะทำการใช้ KMeans แล้วแบ่งข้อมูลออกเป็น 3 กลุ่ม กลุ่มที่ 1 คือ Most Safe for Women กลุ่มที่ 2 คือ Average Safe for Women กลุ่มที่ 3 คือ Least Safe for Women
        * แสดงผลให้ผู้ใช้เลือกข้อมูลของประเทศต่าง ๆ ผ่านกราฟ ที่ซึ่งแต่ละกลุ่มแต่ละประเทศถูกจัดอยู่
        * กราฟใช้ Scatter plot เพื่อให้เห็นการกระจายของข้อมูลประเทศต่าง ๆ ใช้ค่า WomenPeaceAndSecurityIndex_Score_2023 และ WomensDangerIndexWDI_TotalScore_2019 เป็นแกน X และ Y และบอกชื่อแต่ละประเทศในกราฟ
        * สีที่ใช้ในกราฟจะแบ่งเป็นตามแต่ละกลุ่ม 3 กลุ่มที่มีโดยสีเขียวคือกลุ่มที่1 สีเหลืองคือกลุ่มที่2 และสีแดงคือกลุ่มที่3 
        * และประเทศที่ผู้ใช้เลือกจะถูกเน้นด้วยสีnavy และขนาดชื่อที่ใหญ่ขึ้น 
- **Agglomerative Clustering**
    * ทฤษฎี-เป็นอัลกอริธึมการ Clustering ที่ไม่ต้องใช้ Label หรือป้ายกำกับ คือ Unsupervised Learning โดยใช้เทคนิค Hierarchical Clustering ซึ่งจะเริ่มจากการที่แต่ละจุดข้อมูลมีเป็นกลุ่มของตัวเอง และค่อย ๆ รวมกลุ่มกันจนกลายเป็นกลุ่มใหญ่ที่มีความคล้ายคลึงกันจนกว่าจะได้จำนวนกลุ่มที่ต้องการ
    * ขั้นตอนการพัฒนาโมเดล
        * ทำการเตรียมข้อมูลให้พร้อมกับ Agglomerative Model ตามที่บอกไว้ในข้างต้น
        * เลือกจำนวนกลุ่มที่ต้องการให้เป็น 3 กลุ่มเหมือนกับกลุ่มใน KMeans Clustering เพื่อจับกลุ่มประเทศที่มีระดับความปลอดภัยคล้ายกัน
        * ผลที่ได้จะใช้ในการแสดงข้อมูลความอันตรายต่อผู้หญิงในแต่ละประเทศ
        * กราฟใช้ Scatter plot เพื่อให้เห็นการกระจายของข้อมูลประเทศต่าง ๆ ใช้ค่า WomenPeaceAndSecurityIndex_Score_2023 และ WomensDangerIndexWDI_TotalScore_2019 เป็นแกน X และ Y และบอกชื่อแต่ละประเทศในกราฟ
        * สีที่ใช้ในกราฟจะแบ่งเป็นตามแต่ละกลุ่ม 3 กลุ่มและประเทศที่ถูกเลือกก็จะเหมือนกับของ KMeans Clustering
- **SVM Classifier**
    * ทฤษฎี-SVM เป็นอัลกอริธึมในการจำแนกประเภทข้อมูล Supervised Learning จะใช้การหาเส้นขอบเขตการจำแนกที่ดีที่สุด(hyperplane)สามารถแยกข้อมูลออกเป็น 2 กลุ่ม โดยไม่จำเป็นต้องเป็นเส้นตรง
    * ขั้นตอนการพัฒนาโมเดล
        * ข้อมูลแบ่งออกเป็นชุด training และ testing ใช้ train_test_split โดย X_train และ X_test แบ่งเป็นชุดฝึกกับลุดทดสอบ และ y_train และ y_test ค่าที่ได้ คลัสเตอร์ที่ได้จาก KMeans แบ่งเป็นชุดฝึกและชุดทดสอบ 
        * สร้าง SVM Model ใช้ SVC เป็นคลาสที่ใช้จำแนกประเภทใน SVM โดยตั้งค่าพารามิเตอร์ random_state=42 เพื่อให้การฝึกมีผลลัพธ์ที่ทำซ้ำได้
        * Model ฝึกด้วยข้อมูลที่ผ่านการสเกลแล้ว X_train_scaled และ y_train เพื่อให้ SVM จำแนกข้อมูลตามคลัสเตอร์จากใน KMeans หลังฝึกเสร็จ Model จะทำนายผลในชุดข้อมูล X_test_scaled ซึ่งใช้คำสั่ง svm.predict(X_test_scaled)
        * ใช้ SVM ทำนายความไม่เท่าเทียมทางเพศที่เกิดขึ้นกับผู้หญิงในแต่ละประเทศโดยอิงจากดัชนีประเทศที่มีความเป็นอันตรายต่อผู้หญิงและความไม่เท่าเทียมทางเพศ
        * ประเมินผลของ SVM จะใช้ Classification Report ที่ให้ข้อมูลเกี่ยวกับควาามถูกต้องของข้อมูลการทำนาย มี Precision, Recall, F1-Score, Accuracy ของModel ในแต่ละคลัสเตอร์
        * กราฟใช้ Scatter plot เพื่อโชว์ผลการทำนายของModel ข้อมูลการทำนายจาก SVM จะแสดงเป็นเครื่องหมาย(X) สี blue สำหรับประเทศที่ทำนายโดย SVM ว่าอยู่กลุ่มไหน 
        * กราฟจะมี แกน X เป็น Women’sDangerIndexGenderInequality_2019 และแกน Y เป็น Women’sDangerIndexWDI_TotalScore_2019 และสีที่ใช้ในกราฟจะแบ่งเป็นตามแต่ละกลุ่ม 3 กลุ่มเหมือนกับของ KMeans Clustering
        """)
elif page  == "About Neural Network":
#การเตรียมข้อมูล ทฤษฎีของอัลกอริทึมที่พัฒนา และขั้นตอนการพัฒนาโมเดล
    st.title("About Neural Network")
    st.subheader("Development Plan")
    st.text("ในการทำโมเดลนี้ใช้ 2 Datasets โดย Datasetที่ 1 เกี่ยวกับความเป็นอันตรายและความปลอดภัยของผู้หญิงในประเทศต่าง ๆ และ Datasetที่ 2 เกี่ยวกับการเปลี่ยนแปลงอุณหภูมิของประเทศต่าง ๆ ทั่วโลกปี 1961-2019")
    st.markdown(""" 
**1. Data Preparation**
- เริ่มการอ่านข้อมูลทั้งสองจาก CSV ไฟล์
- ทำความสะอาดข้อมูล Datasetที่ 2 โดยการเติมค่า Missing ในข้อมูลด้วย 0
- ใน Datasetที่ 2 เลือกใช้เฉพาะข้อมูลที่เกี่ยวกับThailand และเลือกประเภทของข้อมูลที่เป็นTemperature change จากปี 1961-2019 โดย เริ่มที่ 1961แล้วบวกเพิ่มขึ้นทีละ 3 ปี 
    * ทำการแปลงข้อมูลจาก wide format เป็น long format ตอนแรกข้อมูล ปีจะแยกเป็นแต่ละคอลัมน์ เช่น Y1961 เป็น 1 คอลัมน์ เราจะทำให้กลายเป็นข้อมูลที่มีแค่ 2 คอลัมน์ คือ ปีและค่าของแต่ละปี คือ คอลัมน์ Year และคอลัมน์ ค่าTemperatureChange 
    * ขั้นตอนต่อไปเราจะเปลี่ยนชื่อปีจาก Y1961 เป็น 1961 เพื่อให้เอาไปใช้คำนวนหรือวิเคราะห์ได้ 
    * ต่อไปลบค่า Missing แล้วให้เหลือแค่ข้อมูลที่เกี่ยวข้อง
    * ต่อไปจับกลุ่มข้อมูลตามปีและคำนวน TemperatureChange เป็นค่าเฉลี่ยตามแต่ละปี
    * แล้วก็ใช้ MinMaxScaler เพื่อปรับสเกลข้อมูลอุณหภูมิให้เหมาะกับการทำงานของ LSTM ให้ค่าอุณหภูมิอยู่ในช่วง 0-1
- เตรียมข้อมูลสำหรับ LSTM โดยใช้ Dataset 2 เพื่อสร้างข้อมูลชุดใหม่ที่ใช้เพื่อทำนาย โดยการใช้ข้อมูลอุณหภูมิในช่วง 6 ปีที่ผ่านมาทำนายอุณหภูมิในปีถัดไป 

**2. The Theory of the Developed Algorithm**
- **LSTM Neural Network (Long Short-Term Memory)**
    * ทฤษฎี-LSTM เป็นหนึ่งในประเภท RNN(Recurrent Neural Network) เพื่อให้เก็บรักษาความจำจากข้อมูลในช่วงเวลาต่าง ๆ ได้ดี เหมาะกับการทำงานกับข้อมูลที่มีลำดับเวลา ซึ่งเหมาะกับ Dataset ที่ 2 ที่นำมาใช้สร้าง Model ที่เกี่ยวกับลำดับปี
    * ขั้นตอนการพัฒนาโมเดล
        * เตรียมข้อมูล LSTM ตามที่บอกไว้ในข้างต้น และเลือกใช้ข้อมูลปีที่มีจนถึงปี 2019 เพื่อใช้ในการฝึก LSTM และเก็บข้อมูลไว้ในตัวแปร data_for_lstm_thailand
        * ต่อไปจะเตรียมชุดข้อมูล โดยสร้างฟังก์ชัน create_dataset เพื่อเตียมชุดข้อมูล แบ่งข้อมูลให้เป็น X และ y ใช้ฝึกและทดสอบModel
        * แล้วก็ใช้ข้อมูลย้อนหลัง 6 ปีผ่านมา(X)เพื่อทำนาย Temperature changeในปีถัดไป(y)
        * หลังจากเตรียมชุดข้อมูลเสร็จ ต่อไปจะปรับแบบข้อมูล X ให้เหมาะกับตอนที่จะป้อนข้อมูลเข้า LSTM แล้ว X มีรูปแบบ[samples, time steps, features]
        * LSTM Model ถูกสร้างโดยใช้ Sequential model ใน Keras และเพิ่ม LSTM ที่มีเป็น 2 ชั้น โดยปรับแต่งค่าจำนวน Units ในแต่ละชั้น ซึ่งชั้นแรกมี Unit และ return_sequences=True เพื่อใส่งออกค่าลำดับของผลลัพธ์ ชั้นที่สองมี Unit และreturn_sequences=False เพราะไม่ต้องการผลลัพธ์ลำดับที่สอง
        * เพิ่มชั้น Dense ที่มี units = 1 เพื่อให้เป็นค่าผลลัพธ์การทำนายTemperature Change
        * ต่อไปทำการ complie model จะใช้ Adam Optimizer เพื่อปรับค่าพารามิเตอร์ของmodelระหว่างการฝึกและ mean squared error loss เพื่อคิดคำนวนการ loss 
        * Model ถูกฝึกในช่วงจำนวนรอบ 10 epoch บนชุดข้อมูลทั้งหมด ใช้ข้อมูลปีที่ผ่านมาเพื่อทำนายอุณหภูมิในอนาต (ปี 2020-2029) และใช้จำนวนตัวอย่างข้อมูลที่ใช้ในการฝึกแต่ละรอบ ข้อมูล 32 ตัวอย่าง (batch)
        * ใช้โมเดลที่ฝึกเสร็จแล้วเพื่อทำนายการเปลี่ยนแปลงอุณหภูมิในอนาคต (สำหรับปี 2020 ถึง 2029) ใช้ Forecasting Future Data โดยใช้ข้อมูลในอดีต Historical Data เพื่อทำนายค่าที่จะเกิดขึ้นในอนาคต โดยใช้ข้อมูล TemperatureChangeในปี 1961-2019 เพื่ออัปเดตข้อมูลการทำนายในแต่ละปีตามลำดับ หรือก็คือใช้ข้อมูลชุดสุดท้ายของชุดฝึก(last_sequence) เพื่อทำนายค่าTemperature Changeในปีต่อไป แล้วใช้ค่าที่ทำนายได้ในแต่ละปีในการอัปเดตข้อมูลลำดับถัดไป เพื่อทำนายในปีต่อไป แล้วก็ย้อนปรับขนาดเพื่อให้ได้อุณหภูมิจริง ๆ 
        """)
elif page == "Machine Learning Demo":
    st.title("Machine Learning Demo")
    data = pd.read_csv("dataset/most-dangerous-countries-for-women-2024.csv", encoding='ISO-8859-1') 

    # Data cleaning
    data.fillna(-1, inplace=True)

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
#Elbow Method
    wcss =[]
    for i in range(1, 15):
        kmeans = KMeans(n_clusters = i, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1, 15), wcss, marker='o', linestyle='--')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    ax.set_title('Elbow Method for Optimal k')
    st.pyplot(fig) 

# KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    used_data['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_labels_kmeans = {0: 'Average Safety for Women', 1: 'Least Safe for Women', 2: 'Most Safe for Women'}
    used_data['KMeans_Cluster_Label'] = used_data['KMeans_Cluster'].map(cluster_labels_kmeans)
    st.subheader("K-Means Clustering")
# โชว์KMeans Clusters
    st.write("KMeans Clustering Results")
    st.write(used_data[['country', 'KMeans_Cluster_Label']])
    cluster_colors = {0: 'yellow', 1: 'red', 2: 'green'}

# กล่องเลือกประเทศที่อยากดูโดยเฉพาะ
    country_list = used_data['country'].unique()
    selected_country = st.selectbox("Select a Country", country_list)
    selected_country_data = used_data[used_data['country'] == selected_country]
    st.subheader(f"Data for {selected_country}")
    st.write("KMeans Cluster Label:", selected_country_data['KMeans_Cluster_Label'].values[0])

# กราฟKMeans
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(used_data['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023'],
            used_data['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'],
            c=used_data['KMeans_Cluster'].map(cluster_colors), alpha=0.6)
    ax.scatter(selected_country_data['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023'],
            selected_country_data['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'],
            c='white', s=200, label=f"{selected_country} (Selected)", edgecolor='navy', linewidth=2)
    for i, row in used_data.iterrows():
        plt.scatter(row['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023'],
            row['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'],
            c=cluster_colors[row['KMeans_Cluster']], s=150, alpha=0.6)
    for i, country in enumerate(used_data['country']):
        if country == selected_country:
            plt.text(used_data['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023'].iloc[i]+0.002,
                 used_data['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'].iloc[i]+2,
                 country, fontsize=8, ha='left', color='navy', weight='bold')
        else:
            plt.text(used_data['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023'].iloc[i]+0.002,
                 used_data['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'].iloc[i]+2,
                 country, fontsize=6, ha='left', color='black')
    ax.set_title('KMeans Clusters for Women Safety')
    ax.set_xlabel('WomenPeaceAndSecurityIndex_Score_2023')
    ax.set_ylabel('Women’sDangerIndexWDI_TotalScore_2019')
    legend_labels = ['Average Safety for Women', 'Least Safe for Women', 'Most Safe for Women']
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in cluster_colors.values()]
    ax.legend(handles, legend_labels, title="Cluster Labels")
    st.pyplot(fig)

# Agglomerative Clustering
    agg_clust = AgglomerativeClustering(n_clusters=3)
    used_data['Agglomerative_Cluster'] = agg_clust.fit_predict(X_scaled)  
    cluster_labels_agg = {0: 'Average Safety for Women', 1: 'Most Safe for Women', 2: 'Least Safe for Women'}
    used_data['Agglomerative_Cluster_Label'] = used_data['Agglomerative_Cluster'].map(cluster_labels_agg)
    st.subheader("Agglomerative Clustering")
# โชว์AGG C
    st.write("Agglomerative Clustering Results")
    st.write(used_data[['country', 'Agglomerative_Cluster_Label']])
# โชว์กลุ่มที่ได้จากการเลือกประเทศจากข้างบน
    selected_country_data = used_data[used_data['country'] == selected_country]
    st.write("Agglomerative Cluster Label:", selected_country_data['Agglomerative_Cluster_Label'].values[0])

    color_map_agg = {0: 'yellow', 1:'green', 2:'red'}
# กราฟAgglomerative Clustering
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(used_data['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023'],
            used_data['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'],
            c=used_data['Agglomerative_Cluster'].map(color_map_agg), alpha=0.6)
    ax.scatter(selected_country_data['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023'],
            selected_country_data['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'],
            c='white', s=200, label=f"{selected_country} (Selected)", edgecolor='navy', linewidth=2)
    for i, row in used_data.iterrows():
        ax.scatter(row['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023'],
                row['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'],
                c=color_map_agg[row['Agglomerative_Cluster']], s=150, alpha=0.6)
    for i, country in enumerate(used_data['country']):
        if country == selected_country:
            plt.text(used_data['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023'].iloc[i]+0.002,
                 used_data['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'].iloc[i]+2,
                 country, fontsize=8, ha='left', color='navy', weight='bold')
        else:
            plt.text(used_data['MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023'].iloc[i]+0.002,
                 used_data['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'].iloc[i]+2,
                 country, fontsize=6, ha='left', color='black')
    ax.set_title('Agglomerative Clustering for Women Safety')
    ax.set_xlabel('WomenPeaceAndSecurityIndex_Score_2023')
    ax.set_ylabel('Women’sDangerIndexWDI_TotalScore_2019')
    legend_labels = ['Average Safety for Women', 'Least Safe for Women', 'Most Safe for Women']
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in cluster_colors.values()]
    ax.legend(handles, legend_labels, title="Cluster Labels")
    st.pyplot(fig)

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
    y_pred_svm = svm.predict(X_test_scaled)
#แสดง reportเกี่ยวกับ Recall, F1-Score,...
    st.write("SVM Classification Report")
    st.text(classification_report(y_test, y_pred_svm))
    st.subheader("SVM Classifier")
#กราฟSVM
    color_map_svm = {0: 'yellow', 1:'green', 2:'red'}
    fig, ax = plt.subplots(figsize=(14,11))
    for i, row in used_data.iterrows():
        ax.scatter(row['MostDangerousCountriesForWomen_WDIGenderInequality_2019'],
                row['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'],
                c=color_map_svm[row['Agglomerative_Cluster']], s=150, alpha=0.6)
    for i, country in enumerate(used_data['country']):
        plt.text(used_data['MostDangerousCountriesForWomen_WDIGenderInequality_2019'].iloc[i],
             used_data['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'].iloc[i]+2,
             country, fontsize=6, ha='left')
    for i, pred in enumerate(y_pred_svm):
        plt.scatter(X_test.iloc[i]['MostDangerousCountriesForWomen_WDIGenderInequality_2019'],
                X_test.iloc[i]['MostDangerousCountriesForWomen_WomensDangerIndexWDI_TotalScore_2019'],
                c='blue', marker='x', s=150, label="Predicted" if i == 0 else " ")

    ax.set_xlabel('Women’sDangerIndexGenderInequality_2019')
    ax.set_ylabel('Women’sDangerIndexWDI_TotalScore_2019')
    ax.set_title('SVM Predictions for Women Gender Inequality 2019')
    st.pyplot(fig)

elif page == "Neural Network Demo":
    st.title("Neural Network Demo")
#Neural Network(LSTM-tensorflow/keras)
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

# ทำนายไปอีก 10 ปี
    future_years = np.arange(2020, 2030) 
    future_data = []
#ข้อมูลชุดสุดท้ายของชุดฝึก
    last_sequence = data_for_lstm_thailand.iloc[-time_step:, 1].values  

# ข้อมูลในอนาคต
    for year in future_years:
    # ที่ทำนายแต่ละปี
        last_sequence_input = last_sequence.reshape((1, time_step, 1))
        next_pred = model.predict(last_sequence_input)
        future_data.append(next_pred[0, 0]) 
    # เลื่อนข้อมูลสำหรับทำนายปีต่อไป
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred[0, 0]

# ย้อนขนาดเพื่อให้ได้อุณหภูมิจริง ๆ 
    future_data = scaler.inverse_transform(np.array(future_data).reshape(-1, 1))
# รวมปีก่อนหน้านั้นกับปีในอนาคต
    all_years = np.concatenate([temperature_data_thailand_yearly['Year'], future_years])
    all_data = np.concatenate([temperature_data_thailand_yearly['TemperatureChange'], future_data.flatten()])
    st.subheader("LSTM Neural Network")
    st.title("Thailand Average Yearly Temperature Change (1961-2030)")
    st.subheader("Choose a Year to View Temperature Change")
# กล่องเลือกปีที่จะดู
    selected_year = st.selectbox('Select Year:', all_years)
    if selected_year in future_years:
    # ปีในอนาคต ดูผลทำนายในปีอนาคต
        predicted_temperature = future_data[future_years.tolist().index(selected_year)]
        st.write(f"Predicted temperature change for the year {selected_year}: {predicted_temperature[0]:.2f} °C")
    else:
    # ปีที่มีจริง ๆ ดูข้อมูลปีนั้นๆที่มี
        historical_temperature = temperature_data_thailand_yearly[temperature_data_thailand_yearly['Year'] == selected_year]['TemperatureChange'].values[0]
        st.write(f"Temperature change for the year {selected_year}: {historical_temperature:.2f} °C")

# แสดงค่า Temperature Change กับ Average temperature change ของปีที่เลือก
    historical_data = temperature_data_thailand_yearly[temperature_data_thailand_yearly['Year'] <= selected_year]
    average_temperature_change = historical_data['TemperatureChange'].mean()
    st.write(f"Average temperature change up to the year {selected_year}: {average_temperature_change:.2f} °C")

# กราฟ LSTM 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temperature_data_thailand_yearly['Year'], temperature_data_thailand_yearly['TemperatureChange'], label='Historical Data', marker='o')
    ax.plot(future_years, future_data, label='Predicted Future Data', color='red')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Temperature Change (°C)')
    ax.set_title('Average Yearly Temperature Change in Thailand (1961-2030)')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)



