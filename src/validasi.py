"""
validasi.py
Pengujian end-to-end untuk pipeline.py (Tahap 6).

Yang diuji:
  A. Pipeline berjalan tanpa error, dua kali berturut-turut
  B. Raw dataset tetap ada dan tidak berubah (hash SHA256 sebelum = sesudah)
  C. Processed dataset dihasilkan oleh script dan bisa dibaca Pandas
  D. Output konsisten: hash run ke-1 = hash run ke-2
  E. Jumlah baris/kolom sesuai operasi, missing values & duplikat sesuai keputusan
  F. Scaling dan encoding benar-benar terlihat di output
  G. Perbandingan dengan dataset referensi (hanya sebagai alat periksa)

Cara menjalankan (dari root project):
    python src/validasi.py
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
PIPELINE = SRC_DIR / "pipeline.py"
RAW_PATH = ROOT_DIR / "data" / "raw" / "automobileEDA_dirty_training.csv"
OUT_PATH = ROOT_DIR / "data" / "processed" / "automobileEDA_processed.csv"

# Folder referensi berada di luar project (lihat Tahap 1).
REF_DIR = ROOT_DIR.parent / "reference-datasets"
REF_ORIGINAL = REF_DIR / "automobileEDA.csv"
REF_PROCESSED = REF_DIR / "automobile_processed.csv"

KOLOM_BARU = [
    "engine-size_scaled", "horsepower_scaled", "curb-weight_scaled",
    "body-style_convertible", "body-style_hardtop", "body-style_hatchback",
    "body-style_sedan", "body-style_wagon",
    "drive-wheels_4wd", "drive-wheels_awd", "drive-wheels_fwd", "drive-wheels_rwd",
    "horsepower-binned_encoded",
]

hasil_tes = []


def cek(nama, lulus, detail=""):
    """Mencatat dan mencetak hasil satu pengujian."""
    hasil_tes.append(bool(lulus))
    status = "LULUS" if lulus else "GAGAL"
    print(f"[{status}] {nama}" + (f"  ->  {detail}" if detail else ""))


def judul(teks):
    print("\n" + "=" * 70 + f"\n{teks}\n" + "=" * 70)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def jalankan_pipeline():
    """Menjalankan pipeline.py sebagai proses terpisah, sama seperti dijalankan
    manual dari terminal. Output terminalnya disembunyikan agar ringkas."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proses = subprocess.run(
        [sys.executable, str(PIPELINE)], cwd=ROOT_DIR,
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    if proses.returncode != 0:
        print(proses.stdout[-2000:])
        print(proses.stderr[-2000:])
    return proses.returncode


def main():
    # ------------------------------------------------------------------
    judul("A-D. EKSEKUSI, RAW DATASET, DAN KONSISTENSI")
    cek("Raw dataset tersedia", RAW_PATH.exists(), str(RAW_PATH))
    if not RAW_PATH.exists():
        sys.exit(1)
    hash_raw_awal = sha256(RAW_PATH)

    # Hapus output lama agar terbukti file benar-benar dibuat oleh script.
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    cek("Output lama dihapus sebelum pengujian", not OUT_PATH.exists())

    kode1 = jalankan_pipeline()
    cek("Run ke-1 berjalan tanpa error", kode1 == 0, f"exit code {kode1}")
    cek("Processed CSV dibuat oleh script", OUT_PATH.exists(), str(OUT_PATH))
    if kode1 != 0 or not OUT_PATH.exists():
        sys.exit(1)
    hash_out_1 = sha256(OUT_PATH)

    kode2 = jalankan_pipeline()
    cek("Run ke-2 berjalan tanpa error", kode2 == 0, f"exit code {kode2}")
    hash_out_2 = sha256(OUT_PATH)
    hash_raw_akhir = sha256(RAW_PATH)

    cek("Raw dataset tidak berubah (hash sama)", hash_raw_awal == hash_raw_akhir,
        hash_raw_akhir[:16] + "...")
    cek("Output konsisten antar-run (hash sama)", hash_out_1 == hash_out_2,
        hash_out_2[:16] + "...")
    print(f"\nSHA256 raw    : {hash_raw_akhir}")
    print(f"SHA256 output : {hash_out_2}")

    # ------------------------------------------------------------------
    judul("E. ISI PROCESSED DATASET")
    raw = pd.read_csv(RAW_PATH)
    out = pd.read_csv(OUT_PATH)
    cek("Output bisa dibaca kembali dengan Pandas", True, f"{out.shape[0]} baris, {out.shape[1]} kolom")

    # Jumlah baris yang diharapkan dihitung langsung dari raw, bukan disalin dari pipeline.
    n_dup = int(raw.duplicated().sum())
    n_price = int(raw.drop_duplicates()["price"].isna().sum())
    baris_harapan = len(raw) - n_dup - n_price
    cek("Jumlah baris sesuai operasi", len(out) == baris_harapan,
        f"{len(raw)} - {n_dup} duplikat - {n_price} price kosong = {baris_harapan}, output {len(out)}")

    kolom_harapan = raw.columns.tolist() + KOLOM_BARU
    hilang = [k for k in kolom_harapan if k not in out.columns]
    cek("Semua kolom asli + 13 kolom baru ada", not hilang and out.shape[1] == len(kolom_harapan),
        f"{out.shape[1]} kolom" + (f", hilang: {hilang}" if hilang else ""))

    missing = out.isna().sum()
    missing = missing[missing > 0]
    cek("Missing values hanya di transaction_date", missing.index.tolist() == ["transaction_date"],
        missing.to_dict())
    cek("Tidak ada duplikat", int(out.duplicated().sum()) == 0, f"{int(out.duplicated().sum())} duplikat")

    tgl = out["transaction_date"].dropna()
    cek("Format tanggal seragam YYYY-MM-DD", bool(tgl.str.match(r"^\d{4}-\d{2}-\d{2}$").all()),
        f"{len(tgl)} tanggal terisi")

    for kol in ["make", "body-style", "drive-wheels", "fuel-system"]:
        s = out[kol]
        seragam = bool((s == s.str.strip().str.lower()).all())
        cek(f"'{kol}' huruf kecil tanpa spasi berlebih", seragam, f"{s.nunique()} kategori")

    # ------------------------------------------------------------------
    judul("F. SCALING DAN ENCODING DI OUTPUT")
    for kol in ["engine-size", "horsepower", "curb-weight"]:
        x = out[kol]
        harapan = (x - x.min()) / (x.max() - x.min())
        selisih = float((out[f"{kol}_scaled"] - harapan).abs().max())
        cek(f"{kol}_scaled = rumus Min-Max", selisih < 1e-9,
            f"rentang {out[f'{kol}_scaled'].min():g}-{out[f'{kol}_scaled'].max():g}, selisih maks {selisih:.1e}")

    for kol in ["body-style", "drive-wheels"]:
        dummies = [k for k in KOLOM_BARU if k.startswith(kol + "_")]
        satu_per_baris = bool((out[dummies].sum(axis=1) == 1).all())
        sesuai = all(((out[kol] == d[len(kol) + 1:]).astype(int) == out[d]).all() for d in dummies)
        cek(f"One-hot '{kol}' benar", satu_per_baris and sesuai, f"{len(dummies)} kolom")

    peta = {"Low": 0, "Medium": 1, "High": 2}
    cek("Ordinal horsepower-binned benar",
        bool((out["horsepower-binned"].map(peta) == out["horsepower-binned_encoded"]).all()),
        out["horsepower-binned_encoded"].value_counts().sort_index().to_dict())

    # ------------------------------------------------------------------
    judul("G. PERBANDINGAN DENGAN DATASET REFERENSI (hanya alat periksa)")
    if not (REF_ORIGINAL.exists() and REF_PROCESSED.exists()):
        print(f"Referensi tidak ditemukan di {REF_DIR}. Bagian G dilewati.")
    else:
        sys.path.insert(0, str(SRC_DIR))
        import pipeline  # dipakai untuk mendapatkan index baris asli (tidak disimpan di CSV)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            df_final = pipeline.transformasi_dataset(
                pipeline.bersihkan_dataset(pipeline.baca_dataset(RAW_PATH)))

        ref = pd.read_csv(REF_ORIGINAL)
        ref_proc = pd.read_csv(REF_PROCESSED)
        print(f"Referensi original : {ref.shape[0]} baris, {ref.shape[1]} kolom")
        print(f"Referensi processed: {ref_proc.shape[0]} baris, {ref_proc.shape[1]} kolom")
        print(f"Output kita        : {out.shape[0]} baris, {out.shape[1]} kolom")

        # Bandingkan nilai per kolom untuk baris yang ada di kedua dataset.
        idx = [i for i in df_final.index if i < len(ref)]
        print(f"\nPerbandingan nilai dengan referensi original ({len(idx)} baris yang sama):")
        print(f"    {'kolom':<20} {'beda':>5}  contoh (index: kita vs referensi)")
        for kol in [k for k in ref.columns if k in df_final.columns]:
            a = df_final.loc[idx, kol]
            b = ref.loc[idx, kol]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                sama = ((a - b).abs() < 1e-6) | (a.isna() & b.isna())
            else:
                sama = (a.astype(str).str.lower() == b.astype(str).str.lower()) | (a.isna() & b.isna())
            beda = sama[~sama].index.tolist()
            if beda:
                contoh = ", ".join(f"{i}: {a[i]} vs {b[i]}" for i in beda[:3])
                print(f"    {kol:<20} {len(beda):>5}  {contoh}")
        print("    (kolom lain identik)")

        # Kolom hasil encoding di referensi processed yang juga kita buat.
        sama_nama = [k for k in KOLOM_BARU if k in ref_proc.columns]
        print(f"\nKolom one-hot yang namanya sama dengan referensi processed: {len(sama_nama)} -> {sama_nama}")
        print("Catatan: nilai *_scaled tidak dibandingkan langsung karena referensi memakai")
        print("jumlah baris dan pilihan kolom yang berbeda, sehingga min/max-nya berbeda.")

    # ------------------------------------------------------------------
    judul("RINGKASAN VALIDASI")
    lulus = sum(hasil_tes)
    print(f"{lulus} dari {len(hasil_tes)} pengujian LULUS")
    if lulus != len(hasil_tes):
        print("Ada pengujian yang GAGAL. Periksa baris bertanda [GAGAL] di atas.")
        sys.exit(1)


if __name__ == "__main__":
    main()