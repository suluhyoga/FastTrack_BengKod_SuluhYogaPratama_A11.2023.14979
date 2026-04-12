# 🧠 Deteksi Dini Depresi Mahasiswa  

Aplikasi berbasis web interaktif untuk mendeteksi dini indikasi depresi pada mahasiswa menggunakan model Machine Learning. Proyek ini dikembangkan sebagai pemenuhan program FAST TRACK pada PROGRAM KEGIATAN BENGKEL KODING Data Science di UNIVERSITAS DIAN NUSWANTORO.

## 📌 Deskripsi Proyek  
Kesehatan mental merupakan aspek krusial dalam kehidupan akademik. Depresi pada mahasiswa adalah masalah kesehatan mental yang semakin meningkat, dimana tekanan akademik, tuntutan sosial, serta faktor gaya hidup sangat mempengaruhi kondisi psikologis. Aplikasi ini menganalisis berbagai metrik dari kebiasaan sehari-hari, beban akademik, hingga stres finansial mahasiswa untuk memprediksi status depresi. Model yang digunakan dalam aplikasi ini adalah **Logistic Regression** yang telah melalui tahap *Feature Selection* dan *Hyperparameter Tuning* guna mendapatkan hasil yang ringan, efisien, dan akurat.

## ✨ Fitur Utama  
* **Formulir Assessment Dinamis:** Antarmuka input pengguna yang rapi dan dikelompokkan secara logis.
* **Analisis Real-Time:** Proses penskalaan (*scaling*) dan prediksi instan menggunakan algoritma *machine learning*.
* **Visualisasi Speedometer (Gauge Chart):** Menampilkan probabilitas tingkat risiko psikologis dalam persentase warna (Aman/Waspada/Bahaya).
* **Rapor Parameter (Bar Chart):** Membandingkan secara visual tingkat tekanan akademik, stres finansial, dan kepuasan belajar.
* **Rekomendasi Cerdas:** Memberikan saran dan peringatan dini berdasarkan hasil akhir prediksi.

## 📂 Struktur Direktori  
📦 FastTrack_BengKod_SuluhYogaPratama_A11.2023.14979  
 ┣ 📂 model  
 ┃ ┣ 📜 fitur_pilihan.pkl  
 ┃ ┣ 📜 model_depresi_fasttrack.pkl  
 ┃ ┗ 📜 scaler_fasttrack.pkl  
 ┣ 📜 app.py  
 ┣ 📜 requirements.txt  
 ┣ 📂 notebook  
 ┃ ┣ 📜 SuluhYogaPratama_A11_2023_14979_FastTrack_DS.ipynb  
 ┗ 📜 README.md
