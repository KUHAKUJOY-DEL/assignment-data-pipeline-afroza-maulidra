# Data Preparation & Pipeline Project: Dataset Otomotif

Pipeline ETL berbasis Python (Pandas dan NumPy) yang mengubah dataset otomotif mentah dan "kotor" (`automobileEDA_dirty_training.csv`) menjadi dataset bersih dan tertransformasi (`automobileEDA_processed.csv`) yang siap untuk analisis lanjutan.

> Individual Assignment: **Data Engineering: Pipeline & Preparation**
> Peserta: AFROZA MAULIDRA IFTIKHAR ZUHRI

---

## 1. Tujuan Project

1. Memeriksa kondisi awal dataset dan mengidentifikasi masalah kualitas data berdasarkan bukti.
2. Membersihkan data (missing values, duplikat, format tidak konsisten, nilai tidak wajar) dengan metode yang alasannya dapat dijelaskan.
3. Menerapkan dua transformasi: **Min-Max Scaling** dan **Categorical Encoding**.
4. Menyusun seluruh proses dalam satu pipeline **Extract → Transform → Load** yang bisa dijalankan ulang dengan hasil yang sama.

## 2. Sumber dan Deskripsi Dataset

| File | Peran | Lokasi |
|---|---|---|
| `automobileEDA_dirty_training.csv` | **Input utama** pipeline | `data/raw/` |
| `automobileEDA.csv` | Referensi original, hanya untuk pemeriksaan | di luar repository |
| `automobile_processed.csv` | Referensi hasil bersih, hanya untuk pemeriksaan | di luar repository |

Ketiga file berasal dari `Dataset_Sesi_3.zip` yang diberikan dalam materi. Dataset berisi spesifikasi teknis dan harga mobil. Setiap baris mewakili satu mobil, dengan kolom seperti merek (`make`), tipe bodi (`body-style`), penggerak roda (`drive-wheels`), ukuran mesin (`engine-size`), tenaga (`horsepower`), berat (`curb-weight`), konsumsi bahan bakar, dan harga (`price`). Selain itu ada kolom `transaction_date`.

Dataset referensi **tidak dipakai sebagai sumber output**. Seluruh isi `automobileEDA_processed.csv` dihasilkan oleh `src/pipeline.py`.

## 3. Struktur Folder

```
data-pipeline-assignment/
├── data/
│   ├── raw/
│   │   └── automobileEDA_dirty_training.csv   
│   └── processed/
│       └── automobileEDA_processed.csv        
├── src/
│   ├── pipeline.py                           
│   └── validasi.py                            
├── documentation/
│   ├── data-flow-diagram.png                  
│   └── data-flow-diagram.mmd                  
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes                            
```

## 4. Kondisi Awal Dataset

Hasil fungsi `inspeksi_dataset()` pada file mentah:

| Aspek | Hasil |
|---|---|
| Ukuran | **205 baris, 30 kolom** |
| Kolom numerik | 19 kolom (`int64` / `float64`) |
| Kolom teks | 11 kolom, termasuk `transaction_date` yang tersimpan sebagai teks |
| Total missing values | **17 sel** di 7 kolom |
| Duplikat | **4 baris** (index 201–204 adalah salinan dari 5, 25, 60, 100) |

Missing values per kolom:

| Kolom | Jumlah | Persen |
|---|---|---|
| `stroke` | 4 | 1,95% |
| `horsepower` | 3 | 1,46% |
| `price` | 3 | 1,46% |
| `transaction_date` | 2 | 0,98% |
| `make` | 2 | 0,98% |
| `num-of-doors` | 2 | 0,98% |
| `horsepower-binned` | 1 | 0,49% |

## 5. Masalah Kualitas Data yang Ditemukan

| No | Masalah | Bukti |
|---|---|---|
| 1 | Missing values | 17 sel di 7 kolom (tabel di atas) |
| 2 | Duplikat | 4 baris identik |
| 3 | Format tanggal bercampur | 4 pola di `transaction_date`: `2025-01-01`, `02/01/2025`, `01-03-2025`, `04-Jan-2025` |
| 4 | Tipe data keliru | `transaction_date` tersimpan sebagai teks, bukan tanggal |
| 5 | Penulisan kategori tidak seragam | `make`: `ALFA-ROMERO`, `Audi`, `BMW`, dan 3 nilai dengan spasi berlebih (`dodge  `, `mercury  `, `porsche  `). `body-style`: `SEDAN`, `Sedan`. `drive-wheels`: `AWD`, `RWD`. `fuel-system`: `MPFI`, `Mpfi` |
| 6 | Nilai bertentangan | Satu mobil diesel (mercedes-benz, `diesel`=1, `compression-ratio` 21,5) tercatat `fuel-system` = `Mpfi` (sistem injeksi bensin), padahal mobil diesel lain memakai `idi` |
| 7 | Missing yang punya alasan teknis | Keempat `stroke` kosong adalah mobil mazda bermesin `rotor` (rotary), yang memang tidak memiliki langkah piston |

Hal yang diperiksa tetapi **tidak dianggap masalah**:
- `symboling` memiliki 25 nilai negatif. Ini wajar karena kolom ini adalah skala risiko asuransi (-3 sampai +3).
- Nilai ekstrem berdasarkan metode IQR (misalnya harga mobil mewah) masih masuk akal untuk mobil sungguhan dan tidak ada bukti bahwa nilainya salah, sehingga tidak dihapus.
- Kolom turunan konsisten: `city-L/100km` = 235 / `city-mpg` di semua baris, dan `diesel` + `gas` = 1 di semua baris.

## 6. Metode Cleaning dan Alasannya

Urutan langkah di fungsi `bersihkan_dataset()`:

| Langkah | Masalah | Metode | Alasan |
|---|---|---|---|
| 1 | Spasi dan huruf besar/kecil | `strip()` dan ubah ke huruf kecil pada 9 kolom kategorikal | `Audi` dan `audi` adalah kategori yang sama. Tanpa penyeragaman, encoding akan menghasilkan kolom ganda |
| 2 | Format tanggal | Setiap pola dibaca dengan aturan eksplisit, lalu diubah ke `datetime` | Urutan hari/bulan dibuktikan dari data: pada pola `dd/mm/yyyy`, bagian pertama ada yang > 12; pada pola `mm-dd-yyyy`, bagian kedua ada yang > 12. Aturan eksplisit mencegah Pandas salah menebak |
| 3 | Duplikat | `drop_duplicates()`, kemunculan pertama disimpan | Dilakukan **setelah** teks diseragamkan agar duplikat yang hanya berbeda format ikut terdeteksi, dan **sebelum** imputasi agar baris kembar tidak memengaruhi median/modus |
| 4 | `fuel-system` diesel bernilai `mpfi` | Diganti dengan sistem bahan bakar mayoritas mobil diesel (`idi`) | Dua kolom lain (`diesel`, `compression-ratio`) sama-sama menunjukkan mobil diesel |
| 5a | `price` kosong | **Baris dihapus** | `price` adalah variabel yang paling mungkin dianalisis atau diprediksi. Mengisinya berarti mengarang nilai target. Dampaknya hanya 3 baris |
| 5b–c | `horsepower` kosong | Median **per kelas** `horsepower-binned` (Low 73, Medium 116) | Median keseluruhan (sekitar 95) akan masuk kelas Low, padahal salah satu baris berlabel Medium. Median per kelas menjaga konsistensi dengan label yang ada |
| 5d | `horsepower-binned` kosong | Dihitung ulang dari `horsepower` | Batas kelas dibaca dari data (Low < 102 ≤ Medium < 156 ≤ High) dan cocok untuk 194 dari 194 baris yang lengkap |
| 5e | `num-of-doors` kosong | Modus per `body-style` | Kedua baris adalah sedan, dan sedan di data ini mayoritas berpintu empat |
| 5f | `stroke` kosong | Median (3,29) | Agar baris tetap bisa dipakai. **Catatan:** untuk mesin rotary ini adalah angka pengganti, bukan ukuran fisik |
| 5g | `make` kosong | Diisi `unknown` | Merek tidak bisa dipastikan dari kolom lain |
| 5h | `transaction_date` kosong | **Dibiarkan kosong** | Mengisinya berarti mengarang tanggal transaksi |
| – | `AWD` di `drive-wheels` | Hanya diubah ke huruf kecil, tetap kategori sendiri | AWD tidak persis sama dengan 4WD, dan tidak ada bukti cukup untuk menggabungkannya |

## 7. Jumlah Data Sebelum dan Sesudah Cleaning

| Metrik | Sebelum | Sesudah |
|---|---|---|
| Jumlah baris | 205 | **198** |
| Jumlah kolom | 30 | 30 |
| Total missing values | 17 | **2** (hanya `transaction_date`, disengaja) |
| Duplikat | 4 | 0 |

Rincian pengurangan baris: 205 − **4** duplikat dihapus − **3** baris dengan `price` kosong = **198**.
Duplikat yang ditemukan pada data mentah ada 4, setelah teks diseragamkan tetap 4, dan yang benar-benar dihapus juga 4.

Missing values per kolom sebelum → sesudah: `make` 2 → 0, `num-of-doors` 2 → 0, `stroke` 4 → 0, `horsepower` 3 → 0, `price` 3 → 0, `horsepower-binned` 1 → 0, `transaction_date` 2 → 2.

Jumlah sel yang berubah per kolom (pada baris yang dipertahankan): `transaction_date` 148 (penyeragaman format), `make` 7, `stroke` 4, `horsepower` 3, `num-of-doors` 2, `body-style` 2, `drive-wheels` 2, `fuel-system` 2, `horsepower-binned` 1.
Tipe data yang berubah: `transaction_date` dari teks menjadi `datetime`.

## 8. Metode Transformasi dan Alasan Pemilihan Kolom

Transformasi dilakukan di fungsi `transformasi_dataset()`. **Kolom asli tidak dihapus**. Hasilnya ditambahkan sebagai 13 kolom baru.

### A. Min-Max Scaling

Rumus: `x_scaled = (x − x_min) / (x_max − x_min)`. Hasilnya berada di rentang 0 sampai 1 karena nilai minimum dan maksimum setiap kolom berbeda.

| Kolom | Min | Max | Kolom hasil |
|---|---|---|---|
| `engine-size` | 61 | 326 | `engine-size_scaled` |
| `horsepower` | 48 | 262 | `horsepower_scaled` |
| `curb-weight` | 1488 | 4066 | `curb-weight_scaled` |

Alasan pemilihan: ketiganya ukuran fisik dengan satuan dan rentang yang sangat berbeda (puluhan, ratusan, ribuan), dan tidak ada nilai kosong setelah cleaning. `length` dan `width` tidak dipilih karena sejak awal sudah berskala 0–1 (nilai maksimumnya 1,0).

Fungsi scaling juga menangani nilai konstan (penyebut nol menghasilkan 0,0 dengan peringatan) dan memastikan kolom bertipe numerik.

### B. Categorical Encoding

| Kolom awal | Metode | Kolom hasil | Alasan |
|---|---|---|---|
| `body-style` | One-Hot Encoding | 5 kolom `body-style_*` | Kategori nominal (tidak ada urutan antara sedan dan wagon) |
| `drive-wheels` | One-Hot Encoding | 4 kolom `drive-wheels_*` | Kategori nominal |
| `horsepower-binned` | Ordinal Encoding (Low=0, Medium=1, High=2) | `horsepower-binned_encoded` | Urutannya nyata, karena dibentuk dari batas nilai horsepower |

One-Hot dipilih untuk kategori nominal agar tidak muncul urutan palsu, misalnya sedan=1 dan wagon=2 yang seolah-olah berarti wagon "lebih besar". Kolom `drive-wheels_awd` hanya berisi satu nilai 1, sebagai konsekuensi dari mempertahankan `awd` sebagai kategori sendiri.

## 9. Contoh Nilai Sebelum dan Sesudah Transformasi

Min-Max Scaling (index adalah nomor baris asli):

| Index | engine-size | engine-size_scaled | horsepower | horsepower_scaled | curb-weight | curb-weight_scaled |
|---|---|---|---|---|---|---|
| 17 | 61 | 0,0000 | 48 | 0,0000 | 1488 | 0,0000 |
| 46 | 326 | 1,0000 | 262 | 1,0000 | 3950 | 0,9550 |
| 105 | 152 | 0,3434 | 95 | 0,2196 | 3197 | 0,6629 |

Contoh perhitungan baris 105: (152 − 61) / (326 − 61) = 91 / 265 = 0,3434.

One-Hot Encoding `body-style`:

| body-style | _convertible | _hardtop | _hatchback | _sedan | _wagon |
|---|---|---|---|---|---|
| convertible | 1 | 0 | 0 | 0 | 0 |
| hatchback | 0 | 0 | 1 | 0 | 0 |
| sedan | 0 | 0 | 0 | 1 | 0 |
| wagon | 0 | 0 | 0 | 0 | 1 |
| hardtop | 0 | 1 | 0 | 0 | 0 |

One-Hot Encoding `drive-wheels`:

| drive-wheels | _4wd | _awd | _fwd | _rwd |
|---|---|---|---|---|
| rwd | 0 | 0 | 0 | 1 |
| fwd | 0 | 0 | 1 | 0 |
| 4wd | 1 | 0 | 0 | 0 |
| awd | 0 | 1 | 0 | 0 |

Ordinal Encoding `horsepower-binned`:

| horsepower | horsepower-binned | horsepower-binned_encoded |
|---|---|---|
| 101 | Low | 0 |
| 111 | Medium | 1 |
| 182 | High | 2 |

Distribusi hasil: Low (0) 113 baris, Medium (1) 61 baris, High (2) 24 baris.

> **Catatan:** min dan max untuk scaling dihitung dari seluruh 198 baris. Belum ada pemisahan data latih/uji, sehingga dataset ini **belum diklaim siap untuk pelatihan model**. Jika nanti dipakai untuk machine learning, min/max harus dihitung hanya dari data latih agar tidak terjadi *data leakage*.

## 10. Jumlah Data Sebelum dan Sesudah Pemrosesan

| Tahap | Baris | Kolom |
|---|---|---|
| Raw dataset | 205 | 30 |
| Setelah cleaning | 198 | 30 |
| Processed dataset (setelah transformasi) | **198** | **43** |

## 11. Validasi dan Keterbatasan

Script `src/validasi.py` menjalankan pipeline dua kali dan menguji hasilnya. Hasil eksekusi: **23 dari 23 pengujian lulus**, meliputi:
- Pipeline berjalan tanpa error, dan output benar-benar dibuat oleh script (output lama dihapus sebelum pengujian).
- File mentah tidak berubah. SHA256 sebelum dan sesudah sama: `F5F0791B7F5AEE468916819DDCDABB2364CAD2EDE7E503AC6E76E815B8628113`.
- Output konsisten antar-run. SHA256 output: `A3E1BE88978EBF3FA16F922CDAA10A5A417D801CB0CC6C6A41AEBA14C4CC0D90`.
- Jumlah baris dan kolom sesuai operasi, missing values hanya di `transaction_date`, dan tidak ada duplikat.
- Nilai `_scaled` sama dengan rumus Min-Max, serta one-hot dan ordinal sesuai kategori aslinya.

**Perbandingan dengan dataset referensi** (`automobileEDA.csv`, untuk 198 baris yang sama) memperlihatkan keterbatasan yang perlu dicatat:

1. **Nilai yang salah isi tidak bisa diperbaiki oleh penyeragaman format.** Dari 12 nilai dengan penulisan tidak seragam, 6 hanya salah format, tetapi 6 lainnya juga berbeda isi dari referensi. Hanya 1 yang berhasil diperbaiki (`fuel-system` mobil diesel). Lima lainnya tetap berbeda: `make` baris 30 (audi vs honda) dan 75 (bmw vs mitsubishi), `drive-wheels` baris 12 (awd vs rwd) dan 50 (rwd vs fwd), serta `fuel-system` baris 18 (mpfi vs 2bbl). Nilai-nilai ini **sengaja tidak disalin dari referensi**, karena referensi tidak boleh menjadi sumber output dan dataset kotornya sendiri tidak memberi bukti cukup untuk menentukan nilai yang benar.
2. **Imputasi menghasilkan perkiraan, bukan nilai asli.** `horsepower` hasil imputasi adalah 73, 116, dan 73, sedangkan nilai di referensi 68, 135, dan 88. `num-of-doors` baris 15 diisi `four`, sedangkan referensi mencatat `two`. `make` baris 100 menjadi `unknown`, sedangkan referensi mencatat nissan.
3. `body-style` baris 90 sudah berbeda di data kotor (hatchback vs wagon) tanpa tanda yang bisa mendeteksinya.
4. Referensi menyimpan `stroke` mesin rotary sebagai kosong, sedangkan pipeline ini mengisinya dengan median.
5. Jumlah baris berbeda: 198 di output ini dan 201 di referensi. Selisih 3 baris berasal dari keputusan menghapus baris dengan `price` kosong. Referensi processed (201 baris, 50 kolom) mengisi `price` tersebut dan memakai pilihan encoding yang berbeda.

## 12. Instalasi Dependency

Dibutuhkan Python 3.12 (diuji dengan 3.12.7). Dari root project, di Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` berisi:

```
pandas==3.1.0
numpy==2.5.4
```

## 13. Cara Menjalankan Pipeline

Dari root project dengan virtual environment aktif:

```powershell
python src\pipeline.py
```

Terminal akan menampilkan pesan untuk setiap tahap: `[EXTRACT]`, hasil inspeksi, `[CLEAN]`, `[TRANSFORM]`, `[LOAD]`, dan ringkasan akhir. Folder `data/processed/` dibuat otomatis jika belum ada. Script bisa dijalankan dari folder mana pun karena path dihitung dari lokasi `pipeline.py`.

Untuk menjalankan pengujian:

```powershell
python src\validasi.py
```

## 14. Konsep ETL dalam Project Ini

**ETL** (Extract, Transform, Load) adalah pola untuk memindahkan data dari sumber ke tujuan sambil mengolahnya agar layak dipakai.

| Tahap | Fungsi di `pipeline.py` | Yang dilakukan |
|---|---|---|
| **Extract** | `baca_dataset()` | Membaca CSV mentah ke DataFrame, dengan pesan error jika file tidak ada atau rusak |
| **Transform** | `inspeksi_dataset()` | Memeriksa kondisi awal tanpa mengubah data |
| | `bersihkan_dataset()` | Data cleaning (langkah 1–5h) |
| | `transformasi_dataset()` | Min-Max Scaling dan Categorical Encoding |
| **Load** | `simpan_dataset()` | Menyimpan CSV, menolak menimpa file mentah, dan membaca ulang hasilnya untuk verifikasi |

Fungsi `main()` menjalankan semuanya berurutan dalam satu kali eksekusi. Semua pengolahan dilakukan pada salinan data, sehingga file mentah tidak pernah ditulis ulang.

Diagram alur data tersedia di [`documentation/data-flow-diagram.png`](documentation/data-flow-diagram.png).

## 15. Lokasi Processed Dataset

```
data/processed/automobileEDA_processed.csv
```

Isi file: 198 baris, 43 kolom (30 kolom hasil cleaning dan 13 kolom hasil transformasi). Tanggal disimpan dalam format `YYYY-MM-DD`.