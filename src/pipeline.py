"""
pipeline.py
Pipeline pengolahan dataset otomotif: Extract -> Transform -> Load.

Alur (satu kali eksekusi):
    EXTRACT   : baca_dataset()          -> membaca data/raw/automobileEDA_dirty_training.csv
    TRANSFORM : inspeksi_dataset()      -> memeriksa kondisi awal (tidak mengubah data)
                bersihkan_dataset()     -> data cleaning
                transformasi_dataset()  -> Min-Max Scaling + Categorical Encoding
    LOAD      : simpan_dataset()        -> menulis data/processed/automobileEDA_processed.csv

Cara menjalankan (dari root project):
    python src/pipeline.py

File mentah tidak pernah ditulis ulang. Menjalankan ulang pipeline dengan
input yang sama menghasilkan file output yang sama.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
# __file__ = lokasi file ini (src/pipeline.py). .parent.parent = naik dua
# tingkat ke root project, sehingga path tetap benar dari folder mana pun
# script dijalankan.
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "automobileEDA_dirty_training.csv"
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "automobileEDA_processed.csv"

# Kolom teks dengan nilai unik <= batas ini dianggap kategorikal, sehingga
# seluruh nilai uniknya ditampilkan.
BATAS_KATEGORI = 25

# Teks yang sering dipakai sebagai pengganti data kosong.
TOKEN_KOSONG = {"", "?", "-", "--", "unknown", "none", "null", "n/a", "na", "nan"}

# --- Konfigurasi cleaning (berdasarkan hasil inspeksi Tahap 2) ---

# Kolom kategorikal yang penulisannya diseragamkan menjadi huruf kecil.
# 'horsepower-binned' tidak termasuk karena penulisannya sudah konsisten
# (Low/Medium/High).
KOLOM_KATEGORI = [
    "make", "aspiration", "num-of-doors", "body-style", "drive-wheels",
    "engine-location", "engine-type", "num-of-cylinders", "fuel-system",
]

# Empat format tanggal yang ditemukan di 'transaction_date'.
# Urutan hari/bulan dibuktikan dari data saat inspeksi:
# - format 'dd/mm/yyyy': bagian pertama ada yang > 12, jadi itu hari
# - format 'mm-dd-yyyy': bagian kedua ada yang > 12, jadi itu hari
FORMAT_TANGGAL = {
    r"^\d{4}-\d{2}-\d{2}$": "%Y-%m-%d",     # 2025-01-01
    r"^\d{2}/\d{2}/\d{4}$": "%d/%m/%Y",     # 02/01/2025
    r"^\d{2}-\d{2}-\d{4}$": "%m-%d-%Y",     # 01-03-2025
    r"^\d{2}-[A-Za-z]{3}-\d{4}$": "%d-%b-%Y",  # 04-Jan-2025
}

# --- Konfigurasi transformation ---

# Kolom numerik yang di-scale. Ketiganya ukuran fisik mesin/mobil dengan
# satuan dan rentang sangat berbeda (puluhan, ratusan, ribuan).
# 'length' dan 'width' TIDAK dipilih karena sudah berskala 0-1 sejak awal.
KOLOM_SCALING = ["engine-size", "horsepower", "curb-weight"]

# Kategori nominal (tidak punya urutan) -> One-Hot Encoding.
KOLOM_ONE_HOT = ["body-style", "drive-wheels"]

# Kategori ordinal (punya urutan jelas) -> Ordinal/Label Encoding.
URUTAN_HORSEPOWER = {"Low": 0, "Medium": 1, "High": 2}


def cetak_judul(teks):
    """Mencetak judul bagian agar output terminal mudah dibaca."""
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


# ---------------------------------------------------------------------------
# EXTRACT
# ---------------------------------------------------------------------------
def baca_dataset(path):
    """Membaca CSV mentah menjadi DataFrame. File mentah tidak diubah."""
    print(f"[EXTRACT] Membaca dataset dari: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Pastikan file ada di folder data/raw/ dengan nama yang benar."
        )

    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        raise ValueError(
            f"File ditemukan tetapi gagal dibaca sebagai CSV: {e}\n"
            "Periksa ukuran file apakah wajar, dan pastikan bukan file "
            "metadata macOS (namanya diawali '._')."
        ) from e

    print(f"[EXTRACT] Berhasil dibaca: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


# ---------------------------------------------------------------------------
# TRANSFORM (bagian 1): INSPEKSI
# ---------------------------------------------------------------------------
def inspeksi_dataset(df):
    """
    Memeriksa kondisi awal dataset dan menampilkannya di terminal.
    Fungsi ini HANYA membaca df, tidak mengubahnya.
    Mengembalikan dictionary ringkasan yang bisa dipakai tahap berikutnya.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # 1. Lima baris pertama
    cetak_judul("1. LIMA BARIS PERTAMA")
    print(df.head().to_string())

    # 2. Ukuran dataset
    cetak_judul("2. UKURAN DATASET")
    print(f"Jumlah baris : {df.shape[0]}")
    print(f"Jumlah kolom : {df.shape[1]}")

    # 3. Nama kolom (!r menampilkan tanda kutip, jadi spasi tersembunyi terlihat)
    cetak_judul("3. NAMA KOLOM")
    for i, nama in enumerate(df.columns, start=1):
        print(f"{i:>2}. {nama!r}")

    # 4. Tipe data
    cetak_judul("4. TIPE DATA SETIAP KOLOM")
    print(df.dtypes.to_string())

    # 5. Missing values
    cetak_judul("5. MISSING VALUES PER KOLOM")
    jumlah_missing = df.isna().sum()
    persen_missing = (jumlah_missing / len(df) * 100).round(2)
    tabel_missing = pd.DataFrame(
        {"jumlah_missing": jumlah_missing, "persen": persen_missing}
    )
    print(tabel_missing.to_string())
    total_missing = int(jumlah_missing.sum())
    kolom_missing = jumlah_missing[jumlah_missing > 0].index.tolist()
    print(f"\nTotal sel kosong : {total_missing}")
    print(f"Kolom bermasalah : {kolom_missing if kolom_missing else 'tidak ada'}")

    # 6. Duplicate records
    cetak_judul("6. DUPLICATE RECORDS")
    jumlah_duplikat = int(df.duplicated().sum())
    print(f"Baris duplikat (kemunculan ke-2 dan seterusnya): {jumlah_duplikat}")
    if jumlah_duplikat > 0:
        print("\nContoh baris yang terlibat duplikasi (maks. 6 baris):")
        print(df[df.duplicated(keep=False)].head(6).to_string())

    # 7. Nilai unik kolom teks/kategorikal
    kolom_teks = df.select_dtypes(exclude="number").columns.tolist()
    cetak_judul("7. NILAI UNIK KOLOM TEKS / KATEGORIKAL")
    for kol in kolom_teks:
        n_unik = df[kol].nunique(dropna=True)
        print(f"\n[{kol}] -> {n_unik} nilai unik")
        if n_unik <= BATAS_KATEGORI:
            for nilai, jumlah in df[kol].value_counts(dropna=False).items():
                print(f"    {nilai!r:<25} {jumlah}")
        else:
            contoh = df[kol].dropna().unique()[:5].tolist()
            print(f"    (terlalu banyak untuk dirinci; contoh: {contoh})")

    # 8. Pemindaian otomatis indikasi masalah pada kolom teks
    cetak_judul("8. INDIKASI MASALAH PADA KOLOM TEKS")
    temuan = []
    for kol in kolom_teks:
        s = df[kol].dropna().astype(str)
        if s.empty:
            continue

        # a. Kolom teks yang sebenarnya berisi angka (tipe data keliru)
        angka = pd.to_numeric(s.str.strip(), errors="coerce")
        rasio_angka = angka.notna().mean()
        if rasio_angka >= 0.5:
            penghalang = s[angka.isna()].unique()[:5].tolist()
            temuan.append(
                f"[{kol}] tersimpan sebagai teks, padahal {rasio_angka:.0%} isinya "
                f"angka. Nilai yang menghalangi konversi: {penghalang}"
            )

        # b. Spasi berlebih (di awal, di akhir, atau ganda di tengah)
        bersih = s.str.strip().str.replace(r"\s+", " ", regex=True)
        n_spasi = int((s != bersih).sum())
        if n_spasi > 0:
            temuan.append(f"[{kol}] {n_spasi} nilai memiliki spasi berlebih")

        # c. Penulisan kategori beragam (huruf besar/kecil, spasi, '-' vs '_')
        kunci = (
            s.str.strip()
            .str.lower()
            .str.replace(r"[\s_\-]+", " ", regex=True)
            .str.strip()
        )
        kelompok = s.groupby(kunci).unique()
        for k, daftar in kelompok.items():
            if len(daftar) > 1:
                temuan.append(
                    f"[{kol}] penulisan beragam untuk satu kategori: {daftar.tolist()}"
                )

        # d. Teks pengganti data kosong (misalnya '?')
        n_token = int(s.str.strip().str.lower().isin(TOKEN_KOSONG).sum())
        if n_token > 0:
            temuan.append(f"[{kol}] {n_token} nilai berupa teks pengganti data kosong")

        # e. Pola format berbeda pada kolom dengan banyak nilai unik
        #    (huruf diganti A, angka diganti 9, lalu pola dihitung)
        if s.nunique() > BATAS_KATEGORI:
            pola = (
                s.str.strip()
                .str.replace(r"[A-Za-z]", "A", regex=True)
                .str.replace(r"\d", "9", regex=True)
            )
            hitung_pola = pola.value_counts()
            if len(hitung_pola) > 1:
                temuan.append(
                    f"[{kol}] ada {len(hitung_pola)} pola format berbeda: "
                    f"{hitung_pola.head(5).to_dict()}"
                )

    if temuan:
        for baris in temuan:
            print(f"- {baris}")
    else:
        print("Tidak ada indikasi otomatis. Tetap periksa nilai unik di bagian 7.")

    # 9. Ringkasan kolom numerik
    kolom_angka = df.select_dtypes(include="number").columns.tolist()
    cetak_judul("9. RINGKASAN STATISTIK KOLOM NUMERIK")
    print(df[kolom_angka].describe().T[["count", "mean", "min", "max"]].round(3).to_string())

    negatif = (df[kolom_angka] < 0).sum()
    negatif = negatif[negatif > 0]
    print("\nKolom numerik yang memiliki nilai negatif:")
    if len(negatif) > 0:
        print(negatif.to_string())
        print("(Nilai negatif belum tentu salah; tergantung arti kolomnya.)")
    else:
        print("tidak ada")

    cetak_judul("INSPEKSI SELESAI")
    return {
        "jumlah_baris": int(df.shape[0]),
        "jumlah_kolom": int(df.shape[1]),
        "total_missing": total_missing,
        "jumlah_duplikat": jumlah_duplikat,
    }


# ---------------------------------------------------------------------------
# TRANSFORM (bagian 2): DATA CLEANING
# Setiap fungsi di bawah mengerjakan SATU langkah cleaning, lalu mengembalikan
# DataFrame hasilnya. Semua bekerja pada salinan, bukan pada file mentah.
# ---------------------------------------------------------------------------
def seragamkan_teks(df):
    """Langkah 1: hapus spasi berlebih di semua kolom teks, lalu jadikan
    huruf kecil untuk kolom kategorikal (misalnya 'Audi' -> 'audi')."""
    for kol in df.select_dtypes(exclude="number").columns:
        df[kol] = df[kol].str.strip().str.replace(r"\s+", " ", regex=True)
    for kol in KOLOM_KATEGORI:
        df[kol] = df[kol].str.lower()
    print("[CLEAN] 1. Spasi berlebih dihapus, kategori diseragamkan ke huruf kecil")
    return df


def standarisasi_tanggal(df, kolom="transaction_date"):
    """Langkah 2: ubah 4 format tanggal yang bercampur menjadi satu tipe
    datetime. Setiap format dibaca dengan aturan eksplisit agar tidak ada
    tebakan hari/bulan dari Pandas."""
    teks = df[kolom]
    bagian = []
    for pola, fmt in FORMAT_TANGGAL.items():
        cocok = teks.str.match(pola, na=False)
        bagian.append(pd.to_datetime(teks[cocok], format=fmt))
    hasil = pd.concat(bagian).reindex(df.index)

    # Pengaman: tanggal yang terisi tapi tidak cocok dengan format apa pun
    # tidak boleh diam-diam berubah menjadi kosong.
    gagal = teks.notna() & hasil.isna()
    if gagal.any():
        raise ValueError(
            f"Ada {int(gagal.sum())} tanggal dengan format tidak dikenal: "
            f"{teks[gagal].unique()[:5].tolist()}"
        )

    df[kolom] = hasil
    print(f"[CLEAN] 2. '{kolom}' diubah ke tipe datetime "
          f"({int(hasil.notna().sum())} terbaca, {int(hasil.isna().sum())} memang kosong)")
    return df


def hapus_duplikat(df):
    """Langkah 3: hapus baris yang isinya sama persis, simpan kemunculan
    pertama. Dilakukan SETELAH teks diseragamkan agar duplikat yang hanya
    berbeda spasi/huruf besar ikut terdeteksi, dan SEBELUM pengisian missing
    values agar baris kembar tidak ikut memengaruhi median/modus."""
    ditemukan = int(df.duplicated().sum())
    index_dihapus = df.index[df.duplicated()].tolist()
    df = df.drop_duplicates(keep="first")
    print(f"[CLEAN] 3. Duplikat ditemukan: {ditemukan}, dihapus: {len(index_dihapus)} "
          f"(index asli: {index_dihapus})")
    return df, ditemukan, len(index_dihapus)


def perbaiki_fuel_system_diesel(df):
    """Langkah 4: perbaiki 'fuel-system' yang bertentangan dengan kolom
    'diesel'. Patokannya adalah sistem bahan bakar yang dipakai mayoritas
    mobil diesel di dataset ini sendiri."""
    diesel = df["diesel"] == 1
    sistem_diesel = df.loc[diesel, "fuel-system"].mode()[0]
    salah = diesel & (df["fuel-system"] != sistem_diesel)

    if salah.any():
        print(f"[CLEAN] 4. {int(salah.sum())} mobil diesel memakai fuel-system tidak wajar:")
        print(df.loc[salah, ["make", "fuel-system", "compression-ratio", "diesel", "gas"]].to_string())
        df.loc[salah, "fuel-system"] = sistem_diesel
        print(f"         -> diganti menjadi '{sistem_diesel}'")
    else:
        print("[CLEAN] 4. Tidak ada konflik fuel-system vs diesel")
    return df


def tangani_missing(df):
    """Langkah 5: tangani missing values per kolom dengan metode yang
    dipilih berdasarkan hasil inspeksi. Mengembalikan df dan jumlah baris
    yang dihapus karena 'price' kosong."""

    # 5a. price kosong -> baris dihapus.
    # price adalah nilai yang paling mungkin dianalisis/diprediksi; mengisinya
    # dengan angka tebakan sama saja mengarang jawaban.
    n_price = int(df["price"].isna().sum())
    df = df.dropna(subset=["price"])
    print(f"[CLEAN] 5a. Baris dengan 'price' kosong dihapus: {n_price}")

    # 5b. Batas kelas 'horsepower-binned' dibaca dari data yang lengkap.
    hp = df["horsepower"]
    kelas = df["horsepower-binned"]
    batas_medium = hp[kelas == "Medium"].min()
    batas_high = hp[kelas == "High"].min()

    def tentukan_kelas(nilai_hp):
        return np.where(nilai_hp < batas_medium, "Low",
                        np.where(nilai_hp < batas_high, "Medium", "High"))

    # Cek bahwa aturan batas ini memang cocok dengan label yang sudah ada.
    lengkap = hp.notna() & kelas.notna()
    cocok = int((tentukan_kelas(hp[lengkap]) == kelas[lengkap]).sum())
    print(f"[CLEAN] 5b. Aturan kelas horsepower: Low < {batas_medium:g} <= Medium "
          f"< {batas_high:g} <= High (cocok untuk {cocok} dari {int(lengkap.sum())} baris)")

    # 5c. horsepower kosong -> median dari kelas horsepower-binned yang sama.
    median_per_kelas = df.groupby("horsepower-binned")["horsepower"].median()
    n_hp = int(hp.isna().sum())
    df["horsepower"] = hp.fillna(kelas.map(median_per_kelas))
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())  # cadangan
    print(f"[CLEAN] 5c. 'horsepower' diisi median per kelas: {n_hp} nilai "
          f"(median: {median_per_kelas.to_dict()})")

    # 5d. horsepower-binned kosong -> dihitung ulang dari horsepower.
    kosong = df["horsepower-binned"].isna()
    df.loc[kosong, "horsepower-binned"] = tentukan_kelas(df.loc[kosong, "horsepower"])
    print(f"[CLEAN] 5d. 'horsepower-binned' dihitung ulang dari horsepower: {int(kosong.sum())} nilai")

    # 5e. num-of-doors kosong -> modus (nilai terbanyak) pada body-style yang sama.
    modus_pintu = df.groupby("body-style")["num-of-doors"].agg(lambda s: s.mode()[0])
    n_pintu = int(df["num-of-doors"].isna().sum())
    df["num-of-doors"] = df["num-of-doors"].fillna(df["body-style"].map(modus_pintu))
    print(f"[CLEAN] 5e. 'num-of-doors' diisi modus per body-style: {n_pintu} nilai")

    # 5f. stroke kosong -> median seluruh data (dibahas di README: mesin rotary).
    median_stroke = df["stroke"].median()
    n_stroke = int(df["stroke"].isna().sum())
    df["stroke"] = df["stroke"].fillna(median_stroke)
    print(f"[CLEAN] 5f. 'stroke' diisi median ({median_stroke:g}): {n_stroke} nilai")

    # 5g. make kosong -> 'unknown' (merek tidak bisa dipastikan dari kolom lain).
    n_make = int(df["make"].isna().sum())
    df["make"] = df["make"].fillna("unknown")
    print(f"[CLEAN] 5g. 'make' diisi 'unknown': {n_make} nilai")

    # 5h. transaction_date kosong -> sengaja dibiarkan kosong.
    print(f"[CLEAN] 5h. 'transaction_date' kosong dibiarkan: "
          f"{int(df['transaction_date'].isna().sum())} nilai (tanggal transaksi tidak dikarang)")
    return df, n_price


def hitung_perubahan(df_sebelum, df_sesudah):
    """Menghitung jumlah sel yang berubah per kolom, hanya pada baris yang
    masih ada setelah cleaning. Nilai kosong yang tetap kosong dianggap sama."""
    sebelum = df_sebelum.loc[df_sesudah.index]
    perubahan = {}
    for kol in df_sesudah.columns:
        a = sebelum[kol]
        b = df_sesudah[kol]
        if pd.api.types.is_datetime64_any_dtype(b):
            b = b.dt.strftime("%Y-%m-%d")
        sama = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
        n = int((~sama).sum())
        if n > 0:
            perubahan[kol] = n
    return perubahan


def bersihkan_dataset(df_raw):
    """Menjalankan seluruh langkah cleaning secara berurutan dan mencetak
    laporan sebelum-sesudah. df_raw tidak diubah karena kita bekerja pada
    salinannya (.copy())."""
    cetak_judul("DATA CLEANING")
    df = df_raw.copy()

    df = seragamkan_teks(df)
    df = standarisasi_tanggal(df)
    df, dup_ditemukan, dup_dihapus = hapus_duplikat(df)
    df = perbaiki_fuel_system_diesel(df)
    df, baris_price_dihapus = tangani_missing(df)

    # ----- Laporan sebelum vs sesudah -----
    cetak_judul("LAPORAN CLEANING")
    print(f"Jumlah baris sebelum cleaning : {len(df_raw)}")
    print(f"Duplikat ditemukan (mentah)   : {int(df_raw.duplicated().sum())}")
    print(f"Duplikat ditemukan (setelah teks diseragamkan): {dup_ditemukan}")
    print(f"Duplikat dihapus              : {dup_dihapus}")
    print(f"Baris dihapus (price kosong)  : {baris_price_dihapus}")
    print(f"Jumlah baris sesudah cleaning : {len(df)}")
    print(f"Jumlah kolom sebelum/sesudah  : {df_raw.shape[1]} / {df.shape[1]}")

    missing = pd.DataFrame({
        "sebelum": df_raw.isna().sum(),
        "sesudah": df.isna().sum(),
    })
    missing = missing[(missing["sebelum"] > 0) | (missing["sesudah"] > 0)]
    print("\nMissing values per kolom (hanya kolom yang pernah kosong):")
    print(missing.to_string())
    print(f"Total missing sebelum: {int(df_raw.isna().sum().sum())}, "
          f"sesudah: {int(df.isna().sum().sum())}")

    print("\nJumlah sel yang berubah per kolom (baris yang dipertahankan):")
    for kol, n in hitung_perubahan(df_raw, df).items():
        print(f"    {kol:<20} {n}")

    print("\nTipe data yang berubah:")
    for kol in df.columns:
        if str(df_raw[kol].dtype) != str(df[kol].dtype):
            print(f"    {kol:<20} {df_raw[kol].dtype} -> {df[kol].dtype}")

    print("\nNilai unik kategori setelah cleaning:")
    for kol in ["make", "body-style", "drive-wheels", "fuel-system", "num-of-doors", "horsepower-binned"]:
        print(f"    {kol:<18} {sorted(df[kol].unique().tolist())}")

    print(f"\nDuplikat tersisa: {int(df.duplicated().sum())}")
    return df


# ---------------------------------------------------------------------------
# TRANSFORM (bagian 3): DATA TRANSFORMATION
# Kolom asli TIDAK dihapus; hasil transformasi ditambahkan sebagai kolom baru.
# ---------------------------------------------------------------------------
def min_max_scaling(df, daftar_kolom):
    """Min-Max Scaling: x_scaled = (x - x_min) / (x_max - x_min).
    Hasil disimpan di kolom baru '<nama>_scaled'. Nilai kosong tetap kosong
    (min/max Pandas otomatis mengabaikan NaN)."""
    for kol in daftar_kolom:
        if not pd.api.types.is_numeric_dtype(df[kol]):
            raise ValueError(f"Kolom '{kol}' bukan numerik, tidak bisa di-scale")

        x = df[kol].astype(float)
        x_min, x_max = x.min(), x.max()
        kolom_baru = f"{kol}_scaled"

        if x_max == x_min:
            # Nilai konstan: penyebut nol, jadi semua diberi 0.0
            print(f"[TRANSFORM] PERINGATAN: '{kol}' bernilai konstan, hasil scaling diisi 0.0")
            df[kolom_baru] = np.where(x.notna(), 0.0, np.nan)
        else:
            df[kolom_baru] = (x - x_min) / (x_max - x_min)

        print(f"[TRANSFORM] Min-Max '{kol}': min={x_min:g}, max={x_max:g} -> "
              f"'{kolom_baru}' rentang {df[kolom_baru].min():g} s.d. {df[kolom_baru].max():g}")
    return df


def one_hot_encoding(df, daftar_kolom):
    """One-Hot Encoding untuk kategori nominal: satu kolom 0/1 per kategori.
    Tidak ada kategori yang 'lebih besar' dari yang lain."""
    for kol in daftar_kolom:
        dummies = pd.get_dummies(df[kol], prefix=kol, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        print(f"[TRANSFORM] One-Hot '{kol}' -> {dummies.columns.tolist()}")
    return df


def ordinal_encoding(df, kolom, urutan):
    """Ordinal Encoding untuk kategori yang memang berurutan."""
    kolom_baru = f"{kolom}_encoded"
    df[kolom_baru] = df[kolom].map(urutan)
    tidak_terpetakan = df[kolom].notna() & df[kolom_baru].isna()
    if tidak_terpetakan.any():
        raise ValueError(
            f"Nilai '{kolom}' tidak ada di peta urutan: "
            f"{df.loc[tidak_terpetakan, kolom].unique().tolist()}"
        )
    df[kolom_baru] = df[kolom_baru].astype(int)
    print(f"[TRANSFORM] Ordinal '{kolom}' -> '{kolom_baru}' dengan urutan {urutan}")
    return df


def transformasi_dataset(df_clean):
    """Menjalankan seluruh transformasi dan mencetak tabel sebelum-sesudah.
    df_clean tidak diubah karena kita bekerja pada salinannya."""
    cetak_judul("DATA TRANSFORMATION")
    df = df_clean.copy()
    kolom_awal = df.columns.tolist()

    df = min_max_scaling(df, KOLOM_SCALING)
    df = one_hot_encoding(df, KOLOM_ONE_HOT)
    df = ordinal_encoding(df, "horsepower-binned", URUTAN_HORSEPOWER)

    # ----- Laporan sebelum vs sesudah -----
    cetak_judul("LAPORAN TRANSFORMATION")
    kolom_baru = [k for k in df.columns if k not in kolom_awal]
    print(f"Jumlah kolom sebelum/sesudah : {len(kolom_awal)} / {df.shape[1]}")
    print(f"Jumlah baris sebelum/sesudah : {len(df_clean)} / {len(df)}")
    print(f"Kolom baru ({len(kolom_baru)}): {kolom_baru}")

    # Contoh baris dipilih yang berbeda-beda agar perubahan terlihat jelas:
    # nilai terkecil, terbesar, dan satu baris di tengah.
    hp = df["horsepower"]
    contoh = [hp.idxmin(), hp.idxmax(), (hp - hp.median()).abs().idxmin()]

    print("\nContoh Min-Max Scaling (sebelum -> sesudah):")
    kolom_tampil = []
    for kol in KOLOM_SCALING:
        kolom_tampil += [kol, f"{kol}_scaled"]
    print(df.loc[contoh, kolom_tampil].round(4).to_string())

    for kol in KOLOM_ONE_HOT:
        print(f"\nContoh One-Hot '{kol}':")
        hasil = [k for k in kolom_baru if k.startswith(f"{kol}_")]
        contoh_kat = df.drop_duplicates(subset=kol).index[:5]
        print(df.loc[contoh_kat, [kol] + hasil].to_string())

    print("\nContoh Ordinal 'horsepower-binned':")
    contoh_ord = df.drop_duplicates(subset="horsepower-binned").index
    print(df.loc[contoh_ord, ["horsepower", "horsepower-binned", "horsepower-binned_encoded"]].to_string())

    # ----- Pemeriksaan otomatis -----
    print("\nPemeriksaan:")
    for kol in KOLOM_SCALING:
        s = df[f"{kol}_scaled"]
        print(f"    {kol}_scaled di rentang 0-1: {bool(s.between(0, 1).all())}")
    for kol in KOLOM_ONE_HOT:
        hasil = [k for k in kolom_baru if k.startswith(f"{kol}_")]
        print(f"    One-hot '{kol}' tepat satu nilai 1 per baris: "
              f"{bool((df[hasil].sum(axis=1) == 1).all())}")
    print(f"    Kolom asli tetap ada: {all(k in df.columns for k in kolom_awal)}")
    return df


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
def simpan_dataset(df, path):
    """Menyimpan processed dataset ke CSV. Folder dibuat otomatis jika belum
    ada. Ada pengaman agar file mentah tidak mungkin tertimpa."""
    cetak_judul("LOAD")
    if path.resolve() == RAW_PATH.resolve():
        raise ValueError("Path output sama dengan file mentah. Penyimpanan dibatalkan.")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # index=False: nomor baris Pandas tidak ikut disimpan sebagai kolom.
        # lineterminator="\n": format akhir baris sama di Windows/Linux, jadi
        # hasil run ulang bisa dibandingkan dengan hash.
        df.to_csv(path, index=False, date_format="%Y-%m-%d", lineterminator="\n")
    except PermissionError as e:
        raise PermissionError(
            f"Tidak bisa menulis {path}. Tutup file tersebut jika sedang dibuka "
            "di Excel atau aplikasi lain, lalu jalankan ulang."
        ) from e

    # Verifikasi: file yang baru ditulis dibaca kembali.
    cek = pd.read_csv(path)
    if cek.shape != df.shape:
        raise ValueError(f"Ukuran file tersimpan {cek.shape} tidak sama dengan data {df.shape}")
    print(f"[LOAD] Disimpan ke: {path}")
    print(f"[LOAD] Dibaca ulang untuk verifikasi: {cek.shape[0]} baris, {cek.shape[1]} kolom (cocok)")


# ---------------------------------------------------------------------------
# FUNGSI UTAMA
# ---------------------------------------------------------------------------
def main():
    print("PIPELINE DATA OTOMOTIF: EXTRACT -> TRANSFORM -> LOAD")
    try:
        # EXTRACT
        df_raw = baca_dataset(RAW_PATH)

        # TRANSFORM
        inspeksi_dataset(df_raw)
        df_clean = bersihkan_dataset(df_raw)
        df_final = transformasi_dataset(df_clean)

        # LOAD
        simpan_dataset(df_final, PROCESSED_PATH)
    except (FileNotFoundError, PermissionError, ValueError, KeyError) as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        print("[ERROR] Pipeline dihentikan. File output tidak dianggap valid.")
        sys.exit(1)

    cetak_judul("RINGKASAN PIPELINE")
    print(f"Raw dataset       : {len(df_raw)} baris, {df_raw.shape[1]} kolom")
    print(f"Setelah cleaning  : {len(df_clean)} baris, {df_clean.shape[1]} kolom")
    print(f"Processed dataset : {len(df_final)} baris, {df_final.shape[1]} kolom")
    print(f"Output            : {PROCESSED_PATH}")
    print("\n[SELESAI] Pipeline berhasil dijalankan.")


if __name__ == "__main__":
    main()