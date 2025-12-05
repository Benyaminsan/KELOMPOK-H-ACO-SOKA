# Kelas B Kelompok H

| Nama                      | NRP        |
|---------------------------|------------|
|          Salomo           | 5027221063 |
|       Gilang Raya         | 5027221045 |
|      ALmendo Kambu        | 5027221073 |
|      Nicholas Arya        | 5027231058 |
|    Benjamin Khawarizmi H. | 5027231078 |

# Optimasi Task Scheduling pada Server Cloud
Menggunakan Algoritma **Ant Colony Optimization (ACO)** dan **Stochastic Hill Climbing (SHC)**
Repositori ini berisi implementasi sistem Task Scheduling pada arsitektur multi-server menggunakan dua algoritma optimasi:
- Ant Colony Optimization (ACO) → Algoritma utama yang kami usulkan
- Stochastic Hill Climbing (SHC) → Algoritma baseline pembanding

Proyek ini dikembangkan untuk memenuhi tugas mata kuliah Strategi Optimasi Komputasi Awan (SOKA).

# Pengujian Algoritma Task Scheduler pada Server IT

Repo ini merupakan kode dari server yang digunakan dalam pengujian Task Scheduling pada Server IT serta contoh algoritma scheduler untuk keperluan mata kuliah **Strategi Optimasi Komputasi Awan (SOKA)**

## 📌Algoritma yang Diimplementasikan

**1. Ant Colony Optimization (ACO) – Algoritma Utama**

File: ``aco_scheduler.py``

**Jenis:** Metaheuristik berbasis koloni semut

**Tujuan:** Menemukan assignment (task → VM) yang meminimalkan total cost, makespan, dan load imbalance.

**Karakteristik:**
- Setiap semut membangun solusi berupa mapping tugas → VM.
- Menggunakan pheromone matrix dan heuristic desirability berdasarkan kemampuan VM.
- Pheromone diperbarui pada setiap iterasi (evaporasi + deposit).
- Mampu menemukan solusi mendekati optimum global dengan eksplorasi intensif.

**2. Stochastic Hill Climbing (SHC) – Baseline Pembanding**

File: ``shc_algorithm.py``

**Jenis:** Local Search

**Cara Kerja:**
- Memulai dari solusi acak.
- Perbaikan dilakukan dengan memodifikasi solusi sedikit demi sedikit.
- Berakhir ketika tidak ada perbaikan lokal.

**Kelemahan:**
- Mudah terjebak pada local optimum.
- Tidak memiliki mekanisme eksplorasi global seperti ACO.

## Cara Penggunaan - Dev

1. Install `uv` sebagai dependency manager. Lihat [link berikut](https://docs.astral.sh/uv/getting-started/installation/)

2. Install semua requirement

```bash
uv sync
```

3. Buat file `.env` kemudian isi menggunakan variabel pada `.env.example`. Isi nilai setiap variabel sesuai kebutuhan

```conf
VM1_IP="10.15.42.77"
VM2_IP="10.15.42.78"
VM3_IP="10.15.42.79"
VM4_IP="10.15.42.80"

VM_PORT=5000
```

4. Algoritma pada contoh di sini merupakan algoritma `Stochastic Hill Climbing`.

![shc_algorithm](https://i.sstatic.net/HISbC.png)

5. Untuk menjalankan server, jalankan docker

```bash
docker compose build --no-cache
docker compose up -d
```

6. Inisiasi Dataset untuk scheduler. Buat file `dataset.txt` kemudian isi dengan dataset berupa angka 1 - 10. Berikut adalah contohnya:

```txt
6
5
8
2
10
3
4
4
7
3
9
1
7
9
1
8
2
5
6
10
```

7. Untuk menjalankan scheduler, jalankan file `scheduler.py`. **Jangan lupa menggunakan VPN / Wifi ITS**

```bash
uv run scheduler.py
```

8. Apabila sukses, akan terdapat hasil berupa file `result.csv` dan pada console akan tampil perhitungan parameter untuk kebutuhan analisis.

`result.csv`

<img width="1285" height="667" alt="image" src="https://github.com/user-attachments/assets/84dc7f68-8ddb-4746-9405-5de58efc9cf2" />

`console`

<img width="701" height="433" alt="image" src="https://github.com/user-attachments/assets/f1803e5c-3c8e-4eed-9fa3-ffaad4fcc47c" />

