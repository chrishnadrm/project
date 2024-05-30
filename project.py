import pandas as pd
import streamlit as st
import numpy as np
from scipy.constants import g
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Menampilkan judul
st.title('Menganalisis pengaruh sudut luncuran terhadap jarak tempuh pada game angry birds')
st.write('|by : KELOMPOK 4_FISIKA C 2023|')

# Inisialisasi list kosong untuk menyimpan data
data = {
    'Konstanta ketapel': [], 
    'Massa Batu': [], 
    'Tarikan ketapel': [], 
    'Sudut': [], 
    'Jarak lontaran': []
}

# Memasukkan data
st.success('isi nilai dibawah sesuai data yang diketahui')
st.error('nilai tidak boleh nol')
col1, col2,col3 = st.columns(3)
with col1:
    k = st.number_input('Masukkan konstanta ketapel (N/m)', value=0.0, step=0.1)
        
with col2:
    m = st.number_input('Masukkan massa angry birds (kg)', value=0.0, step=0.1)
        
with col3:
    X = st.number_input('Masukkan tarikan (m)', value=0.0, step=0.1)
if k == 0:
        st.stop()
elif m == 0:
        st.stop()
elif X == 0:
        st.stop()

# Memasukkan jumlah anak
n = st.number_input('banyaknya angry birds:', step=1, value=2)

for i in range(n):
    s = st.number_input(f'sudut luncuran ketapel angry birds {i+1} (dalam derajat)', value=0.0, step=0.1)

#rumus mencari kecepatan awal dan jarak tempuh
    theta = np.radians(s)
    v0 = np.sqrt(k*X**2/m)
    d = (v0**2)*np.sin(2*theta)/g

    data['Konstanta ketapel'].append(k)
    data['Massa Batu'].append(m)
    data['Tarikan ketapel'].append(X)
    data['Sudut'].append(s)
    data['Jarak lontaran'].append(d)

# Membuat dataframe dari data yang dimasukkan
df = pd.DataFrame(data)

# Menampilkan tabel data
st.write('Data yang Dimasukkan:')
st.write(df)

# Memilih variabel independen dan dependen
x_nilai = df['Sudut']
y_nilai = df['Jarak lontaran']

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
if st.button('Tampilkan'):
    # Menampilkan plot regresi
    st.write('Grafik:')
    plt.scatter(x, y, color='blue', label ='data')
    plt.plot(x, model.predict(x), color='red', label='linear regression')
    plt.title('Grafik sudut luncuran ketapel terhadap tempuh')
    plt.xlabel('sudut luncuran')
    plt.ylabel('Jarak lontaran')
    plt.grid()
    plt.legend()
    st.pyplot(plt)

# Menampilkan persamaan dan R squared
    st.write('persamaan linear dan R²:')
    st.write('=================================')
    st.write(persamaan)
    st.write('R²:',r_squared)
    st.write('=================================')