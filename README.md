# Attention-Driven Meta-Learning Project

This repository contains an implementation of meta-learning for few-shot classification using attention mechanisms, specifically focusing on the Convolutional Block Attention Module (CBAM) integrated with ResNet18. The project utilizes the PlantVillage dataset and aims to demonstrate the benefits of attention-driven meta-learning for feature extraction and classification tasks.

## Project Overview

This project consists of three main components:

1. **Model Definition and Training**: Implements the ResNet18 architecture with and without CBAM. Uses Model-Agnostic Meta-Learning (MAML) for training on few-shot learning tasks generated from the PlantVillage dataset.

2. **Evaluation**: Evaluates the performance of the trained models on unseen classes using meta-learning evaluation techniques to compute metrics like accuracy, precision, recall, and F1-score.

3. **Feature Visualization**: Visualizes the feature embeddings from the trained models using t-SNE to understand how well the models cluster different classes.

## Requirements

To run this project, you will need:

- Python 3.x
- PyTorch
- torchvision
- NumPy
- SciPy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- PIL (Pillow)
- `higher` library for MAML implementation
- TensorFlow Datasets (for dataset download)

Install all dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training Models

Train ResNet18 models (with and without CBAM) using the PlantVillage dataset and MAML:

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
    --pretrained
```

### 2. Evaluating Models

Evaluate the trained models on unseen classes using the following command:

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
    --model_checkpoint models/checkpoints/maml_MT10_resnet18_cbam_kshot1.pth
```

### 3. Visualizing Feature Embeddings

Visualize the feature embeddings from trained models using t-SNE with the command:

```bash
python src/utils/visualization.py \
    --dataset_dir data/processed/dataset_MT10/unseen \
    --model resnet18_cbam \
    --model_checkpoint models/checkpoints/maml_resnet18_cbam.pth \
    --n_samples 1000 \
    --output resnet18_cbam_tsne.png
```

## Results

### Model Evaluation Metrics

The evaluation metrics computed include:
- **Accuracy**
- **Precision**
- **Recall**
- **Specificity**
- **F1-Score**

The performance metrics are computed using unseen tasks and reported with 95% confidence intervals.

### Feature Visualization

The feature embeddings are visualized using t-SNE, showing how well the models cluster the different classes. The plots are saved to the specified output file.

## Customizing the Project

- **Model Architecture**: Modify the model architecture in `src/models/model_definition.py`.
- **Training Settings**: Adjust the hyperparameters, such as number of tasks, inner loop learning rate, and meta-learning rate in `src/training/meta_training.py`.
- **Evaluation Metrics**: Modify or add new evaluation metrics in `src/evaluation/evaluate.py`.
- **Data Augmentation**: Update data preprocessing or augmentation steps as needed in the dataset preparation phase.

## Notes

- The dataset should be processed before training by using the script provided.
- Ensure the required CUDA version and GPU drivers are available for training on GPU.

## Troubleshooting

- **State Dict Loading Errors**: If you encounter errors regarding mismatched keys during model loading, make sure the state dict keys match by adjusting the prefixes or using `strict=False` while loading the model.
- **Out of Memory**: For large models, use smaller batch sizes or reduce the number of inner steps to avoid CUDA memory errors.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The PlantVillage dataset used in this project was accessed through TensorFlow Datasets.
- Inspired by research in meta-learning and attention mechanisms in computer vision.

## Contact

If you have any questions or suggestions, please feel free to contact us or raise an issue on the repository.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.
