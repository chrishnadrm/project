import pandas as pd
import streamlit as st
import numpy as np
from scipy.constants import g
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math

# Menampilkan judul
st.title('kalkulator mencari persamaan dan grafik mengenai tarikan ketapel terhadap jarak lemparan')
st.write('|  by : kelompok 4  |')

# Memasukkan nama variabel
col1, col2, col3 = st.columns(3)
with col1:
    pegas = st.text_input('variabel yang diketahui', 'konstanta')
with col2:
    massa = st.text_input('variabel yang diketahui', 'massa batu')
with col3:
    sudut = st.text_input('variabel yang diketahui', 'sudut')

col1, col2 = st.columns(2)
with col1:
    tarikan = st.text_input('sumbu x pada grafik', 'tarikan')
with col2:
    jarak = st.text_input('sumbu y pada grafik', 'jarak')

# Inisialisasi list kosong untuk menyimpan data
data = {pegas: [], massa: [], sudut: [], tarikan: [], jarak: []}

# Memasukkan data
st.success('isi nilai dibawah sesuai data yang diketahui')
st.error('nilai tidak boleh nol')
col1, col2,col3 = st.columns(3)
with col1:
    k = st.number_input(f'Nilai {pegas} (N/m)', value=0.0, step=0.1)
    
with col2:
    m = st.number_input(f'Nilai {massa} (kg)', value=0.0, step=0.1)
    
with col3:
    s = st.number_input(f'Nilai {sudut} (θ)', value=0.0, step=0.1)

# Memasukkan jumlah anak
num_data = st.number_input('banyak anak yang ikut:', min_value=2, step=1, value=2)

for i in range(num_data):
    st.write(f'anak ke-{i+1}:')
    j = st.number_input(f'{tarikan} ketapel anak ke-{i+1} (m):', value=0.0, step=0.1)
    if j == 0:
        st.error('semua nilai tidak boleh nol')
        st.stop()
#rumus mencari kecepatan awal dan jarak tempuh
    theta = np.radians(s)
    F = k*j
    v0 = math.sqrt(2*F/m)
    d = (v0**2)*np.sin(2*theta)/g
    data[pegas].append(k)
    data[massa].append(m)
    data[sudut].append(s)
    data[tarikan].append(j)
    data[jarak].append(d)


# Membuat dataframe dari data yang dimasukkan
df = pd.DataFrame(data)

# Menampilkan tabel data
st.write('Data yang Dimasukkan:')
st.write(df)

# Memilih variabel independen dan dependen

x_nilai = df[tarikan]
y_nilai = df[jarak]

x = np.array(x_nilai)
y = np.array(y_nilai)
x = x.reshape(len(x),1)
y = y.reshape(len(y),1)

# Membuat model regresi linear
model = LinearRegression()
model.fit(x, y)

# Prediksi nilai y berdasarkan model
y_pred = model.predict(x)

# Menghitung R-squared
r_squared = r2_score(y, y_pred)

# Menampilkan persamaan regresi linear
slope = model.coef_[0]
intercept = model.intercept_
persamaan = (f'y = {slope}x + {intercept}')

# Tombol untuk menampilkan plot regresi, persamaan, dan R-squared
if st.button('Tampilkan Grafik, Persamaan, dan R-squared'):
    # Menampilkan plot regresi
    st.write('Grafik:')
    plt.scatter(x, y, color='blue', label ='data')
    plt.plot(x, model.predict(x), color='red', label='linear regression')
    plt.title('grafik tarikan terhadap jarak lontaran batu')
    plt.xlabel(tarikan)
    plt.ylabel(jarak)
    plt.legend()
    st.pyplot(plt)

    # Menampilkan persamaan dan R squared
    st.write('persamaan linear dan R²:')
    st.write('=================================')
    st.write(persamaan)
    st.write('R²:',r_squared)
    st.write('=================================')

    
    if model.coef_[0] > 0:
        naik = (f'keterangan = {tarikan} berbanding lurus terhadap {jarak}')
        st.success(naik)
    elif model.coef_[0] < 0:
        turun = (f'keterangan = {tarikan} berbanding terbalik terhadap {jarak}')
        st.success(turun)
    else:
        konstan = (f'keterangan = {tarikan} tidak mempengaruhi {jarak}')
        st.error(konstan)
