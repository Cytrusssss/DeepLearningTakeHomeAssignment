# Studi Komparatif Model Generatif

Repositori ini berisi implementasi modular untuk Vanilla AE, Sparse AE, VAE,
Vanilla GAN, DCGAN, dan DDPM pada FashionMNIST.

## Struktur

- `main.ipynb`: notebook asli yang tidak dimodifikasi.
- `main_modular.ipynb`: notebook ringkas sebagai runner modular.
- `main.py`: eksekusi pipeline dari terminal.
- `src/config.py`: konfigurasi, seed, dan direktori.
- `src/data.py`: dataset dan pembagian train/validation/test.
- `src/models/`: arsitektur AE, VAE, GAN, dan DDPM.
- `src/training.py`: loop training untuk semua keluarga model.
- `src/evaluation.py`: metrik, klasifikasi, dan latency.
- `src/visualization.py`: pembangkitan figur.

## Menjalankan

```bash
python -m pip install -r requirements.txt
python main.py
```

Untuk uji cepat:

```bash
python main.py --quick --mlp-epochs 1 --ae-epochs 1 --gan-epochs 1 --ddpm-epochs 1
```
