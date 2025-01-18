# Attention-Driven Meta-Learning Project

This repository contains an implementation of meta-learning for few-shot classification using attention mechanisms, specifically **ResNet18** integrated with the **Convolutional Block Attention Module (CBAM)**. The methodology is based on **Model-Agnostic Meta-Learning (MAML)** and is applied to the **PlantVillage** dataset to showcase how attention-driven meta-learning can enhance feature extraction and classification performance, even in scenarios with limited training data.

---

## 1. Project Overview

This project has three main components:

1. **Model Definition and Training**  
   - Implements ResNet18 with and without CBAM  
   - Trains the models using MAML on few-shot tasks derived from the PlantVillage dataset

2. **Evaluation**  
   - Evaluates the trained models on unseen classes to compute metrics such as accuracy, precision, recall, specificity, and F1-score

3. **Feature Visualization**  
   - Uses t-SNE to visualize the embeddings from the trained models, illustrating how well the models cluster different classes

---

## 2. Installation and Requirements

The code is tested on **Python 3.8** with **CUDA 11.1**. GPU usage is highly recommended due to the computational overhead of meta-learning.

**Required Packages:**

- Python 3.x  
- [PyTorch](https://pytorch.org/) (>= 1.9)  
- [torchvision](https://pytorch.org/vision/)  
- NumPy  
- SciPy  
- scikit-learn  
- matplotlib  
- seaborn  
- tqdm  
- PIL (Pillow)  
- [higher](https://github.com/facebookresearch/higher) (for MAML inner-loop optimization)  
- TensorFlow Datasets (for dataset download if using TFDS)

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note**: Verify that your CUDA and GPU drivers are installed if you plan to train on GPU. Training on CPU is possible but significantly slower.

---

## 3. Directory Structure and Data Preparation

Ensure your project directory looks like this (you may adapt it as needed):

```
attention-meta-learning/
├── data/
│   ├── raw/                # Original dataset files
│   ├── processed/
│   │   └── dataset_MT10/   # Preprocessed data for the MT-10 experiment
│   ├── ...                 # Additional datasets can be placed here
├── models/
│   └── checkpoints/        # Directory to save trained model checkpoints
├── src/
│   ├── training/
│   ├── evaluation/
│   ├── utils/
│   ├── models/
├── results/
│   └── figures/            # Directory for saving t-SNE plots and other results
├── requirements.txt
├── README.md
└── ...
```

### 3.1 Downloading the Dataset

The PlantVillage dataset can be accessed through TensorFlow Datasets or manually downloaded from its source and placed under `data/raw/`. If you are using TFDS, the dataset will be automatically downloaded to the default TFDS directory.

### 3.2 Data Preprocessing

Run the provided script to preprocess the dataset for the MT-10 (tomato classes) or MT-6 experiments:

```bash
python src/data_preprocessing.py \
    --input_dir data/raw/plant_village \
    --output_dir data/processed/dataset_MT10 \
    --experiment MT10
```

This will split the dataset into training and unseen sets according to the N-way K-shot experimental setup.

---

## 4. Training the Models

Below is an example command for training ResNet18 with CBAM on the MT-10 experiment using MAML:

```bash
python src/training/meta_training.py \
    --dataset_dir data/processed/dataset_MT10/train \
    --experiment MT10 \
    --model resnet18_cbam \
    --n_way 10 \
    --k_shot 5 \
    --q_queries 15 \
    --meta_batch_size 2 \
    --inner_steps 5 \
    --pretrained \
    --seed 42
```

**Key Arguments:**

- `--dataset_dir`: Path to the preprocessed training data  
- `--model`: Either `resnet18` or `resnet18_cbam`  
- `--n_way`: Number of classes (N-way)  
- `--k_shot`: Number of samples per class (K-shot)  
- `--inner_steps`: Number of inner loop gradient steps for MAML  
- `--seed`: Ensures reproducibility by setting all relevant random seeds (NumPy, PyTorch)

Training logs and checkpoints will be saved automatically in `models/checkpoints/`.

---

## 5. Evaluating the Models

Once training is complete, evaluate your model on unseen classes:

```bash
python src/evaluation/evaluate.py \
    --dataset_dir data/processed/dataset_MT10/unseen \
    --experiment MT10 \
    --model resnet18_cbam \
    --n_way 10 \
    --k_shot 1 \
    --q_queries 15 \
    --num_tasks 600 \
    --inner_steps 5 \
    --inner_lr 0.01 \
    --model_checkpoint models/checkpoints/maml_MT10_resnet18_cbam_kshot1.pth \
    --seed 42
```

**Key Arguments:**

- `--num_tasks`: Number of few-shot tasks to evaluate  
- `--inner_lr`: Learning rate for the inner loop during adaptation  
- `--model_checkpoint`: Path to the trained model checkpoint  
- `--seed`: Again ensures reproducibility during evaluation

The script outputs metrics like accuracy, precision, recall, specificity, and F1-score with 95% confidence intervals.

---

## 6. Visualizing Feature Embeddings

To visualize feature embeddings via t-SNE:

```bash
python src/utils/visualization.py \
    --dataset_dir data/processed/dataset_MT10/unseen \
    --model resnet18_cbam \
    --model_checkpoint models/checkpoints/maml_MT10_resnet18_cbam_kshot5.pth \
    --n_samples 1000 \
    --output resnet18_cbam_tsne.png
```

The resulting plot is saved in `results/figures/`.  

---

## 7. Reproducibility

- **Seeds**: We set a consistent seed (default 42) for all relevant packages (NumPy, PyTorch) to ensure reproducible results. If you wish to replicate our exact numbers, make sure to use the same seeds.  
- **Hardware**: All experiments were conducted on an **NVIDIA GeForce RTX 2080 Ti** GPU, Intel Core i9 CPU, and 32GB of RAM.  
- **Logs and Checkpoints**: We recommend keeping logs and checkpoints for future reference or for hyperparameter tuning without re-training from scratch.

---

## 8. Results

### 8.1 Model Evaluation Metrics

**Tables** in the manuscript (Tables 1 and 2) provide metrics such as accuracy, precision, recall, specificity, and F1-score for different \( k \)-shot settings. Detailed numerical results are saved in `results/evaluation_logs/`.

### 8.2 Feature Visualization

t-SNE plots show how well the models cluster different classes in 2D space, indicating the effectiveness of the attention mechanism for feature separation in the few-shot setting.

---

## 9. Customizing the Project

1. **Model Architecture**: Edit `src/models/model_definition.py` to modify or add new layers.  
2. **Training Settings**: Adjust hyperparameters in `src/training/meta_training.py`, e.g., `--meta_batch_size`, `--inner_lr`, `--outer_lr`, etc.  
3. **Evaluation Metrics**: Add or revise metrics in `src/evaluation/evaluate.py`.  
4. **Data Augmentation**: Update augmentations in the data preprocessing script or in `src/training/meta_training.py`.

---

## 10. Troubleshooting

1. **Mismatched Keys in State Dict**: If model loading fails, set `strict=False` or adapt your checkpoint keys to match your model layers.  
2. **CUDA Out of Memory**: Reduce `--meta_batch_size` or the number of inner steps. Mixed-precision training can also help.  
3. **Dataset Structure Issues**: Check if the dataset folders are named correctly (`train`, `unseen`, etc.) and contain images in the expected format.

---

## 11. License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

---

## 12. Acknowledgments

- The PlantVillage dataset was accessed through TensorFlow Datasets or manually downloaded from the original source.
- This research was inspired by advancements in meta-learning, attention mechanisms, and computer vision.

---

## 13. Contact

If you have questions or suggestions, please [open an issue](../../issues) or contact the authors directly.

---

## 14. Contributing

Contributions are welcome. Please fork this repository, make the desired changes, and submit a pull request.
