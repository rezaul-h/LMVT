# LMVT: Hybrid Vision Transformer for Lung Cancer Classification

This repository implements **LMVT** (“Lung MobileViT”), a lightweight hybrid CNN–Transformer architecture with attention mechanisms and texture-based features, for efficient and explainable lung cancer diagnosis on CT and histopathology images. It supports stratified 10-fold cross-validation, ablation studies, baseline comparisons, and Grad-CAM interpretability.

---

## ⭐️ Key Features

- **Hybrid Architecture**  
  Combines MobileViT backbone + CBAM spatial/channel attention + MHSA global context + SGLDM texture features.

- **Modular Design**  
  Clean separation of configuration, data pipeline, feature extraction, model code, training, evaluation, and explainability.

- **Stratified 10-Fold CV**  
  Ensures robust performance estimation across IQ-OTH/NCCD, LC25000, and external LIDC-IDRI datasets.

- **Ablation & Baselines**  
  Easy scripts to disable CBAM, SGLDM, MHSA and to compare against popular CNNs/ViTs.

- **Explainable AI**  
  Grad-CAM implementation + quantitative CAM metrics (activation area, noise ratio, edge density).

---

## 📦 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/lmvt-lung-cancer.git
   cd lmvt-lung-cancer
   ```

2. **Create environment & install dependencies**  
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Download datasets**  
   - IQ-OTH/NCCD → `./data/iqoth-nccd/`  
   - LC25000 → `./data/lc25000/`  
   - LIDC-IDRI (optional for external validation) → `./data/lidc-idri/`

   Update paths in `config/config.yaml` as needed.

---



## 🚀 Usage

### 1. Train only
```bash
python main.py --config config/config.yaml --mode train
```

### 2. Evaluate only
```bash
python main.py --config config/config.yaml --mode eval                --checkpoint logs/best_model_fold0.pth --fold 0
```

### 3. Train & Evaluate sequentially
```bash
python main.py --config config/config.yaml --mode both                --checkpoint logs/best_model_fold2.pth --fold 2
```

### 4. Run Ablations
```bash
python experiments/ablation_runner.py
```

### 5. Run Baseline Comparisons
```bash
python experiments/baseline_runner.py
```


---

## 🤝 Contributing

1. Fork the repo  
2. Create your branch (`git checkout -b feat/YourFeature`)  
3. Commit (`git commit -m 'Add feature'`)  
4. Push (`git push origin feat/YourFeature`)  
5. Open a Pull Request

---

## 📝 License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

**© 2025** Rezaul Haque / East West University  
