# AI_Tubes2_AiKamuMah
Repository ini dibuat untuk memenuhi Tugas Besar 2 Mata Kuliah Inteligensi Artifisial IF3170 2025/2026. Pada Tugas Besar 2 ini, mahasiswa ditugaskan untuk membuat model Decision Tree Learning, Logistic Regression, dan Support Vector Machine untuk memprediksi  apakah seorang siswa lulus/dropout berdasarkan data performa mahasiswa selama 2 semester pertama

### Setup and Run

#### Setup Virtual Environment
```bash
# Windows
python -m venv venv
.venv\Scripts\activate
pip install -r requirements.txt

# Mac/Linux
python3 -m venv venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Struktur Folder
```
AI_Tubes2_AiKamuMah/
│
├── data/                   # Tempat menyimpan dataset (raw dan processed)
│   ├── raw/                # Dataset asli (jangan diubah)
│   └── processed/          # Dataset hasil cleaning/preprocessing
│
├── doc/                    # Laporan Tugas Besar
│   └── Laporan_Tugas_Besar_2.pdf
│
├── models/                 # Folder untuk menyimpan model yang sudah dilatih (.pkl/.txt)
│   ├── dtl_model.pkl
│   ├── logreg_model.pkl
│   └── svm_model.pkl
│
├── src/                    # Source code implementasi "From Scratch"
│   ├── __init__.py
│   ├── dtl.py              # Implementasi ID3/C4.5/CART
│   ├── logistic_regression.py # Implementasi Logistic Regression
│   ├── svm.py              # Implementasi SVM & Multiclass logic 
│   ├── metrics.py          # Fungsi perhitungan akurasi/F1-score manual
│   └── utils.py            # Helper functions (Save/Load, Visualization Bonus)
│
├── .gitignore              # File untuk ignore folder data/ dan models/ (biar repo gak berat)
├── notebook.ipynb          # Notebook utama untuk EDA dan training
├── README.md               # Dokumentasi cara run & pembagian tugas [cite: 114]
└── requirements.txt        # List library (numpy, pandas, scikit-learn, matplotlib)
```
### Pembagian Tugas
| Name                          | Contribution                        |
|-------------------------------|-------------------------------------|
| Andi Farhan Hidayat           | Model Optimization & Comparison     |
| Ahmad Syafiq                  | Data Preprocessing & Cleaning       |
| Andri Nurdianto               | SVM                                 |
| Rafael Marchel Darma Wijaya   | DTL                                 |
| Muhammad Kinan Arkansyaddad   | Logistic Regression                 |
