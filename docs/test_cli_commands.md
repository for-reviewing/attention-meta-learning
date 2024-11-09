## **Evaluation Commands for ResNet18 + CBAM**

### **MT-10 Experiment**

#### **k_shot = 1**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT10/unseen --experiment MT10 --model resnet18_cbam --n_way 10 --k_shot 1 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT10_resnet18_cbam_kshot1.pth
```

#### **k_shot = 5**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT10/unseen --experiment MT10 --model resnet18_cbam --n_way 10 --k_shot 5 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT10_resnet18_cbam_kshot5.pth
```

#### **k_shot = 10**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT10/unseen --experiment MT10 --model resnet18_cbam --n_way 10 --k_shot 10 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT10_resnet18_cbam_kshot10.pth
```

#### **k_shot = 15**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT10/unseen --experiment MT10 --model resnet18_cbam --n_way 10 --k_shot 15 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT10_resnet18_cbam_kshot15.pth
```

#### **k_shot = 20**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT10/unseen --experiment MT10 --model resnet18_cbam --n_way 10 --k_shot 20 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT10_resnet18_cbam_kshot20.pth
```

### **MT-6 Experiment**

#### **k_shot = 1**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT6/unseen --experiment MT6 --model resnet18_cbam --n_way 6 --k_shot 1 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT6_resnet18_cbam_kshot1.pth
```

#### **k_shot = 5**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT6/unseen --experiment MT6 --model resnet18_cbam --n_way 6 --k_shot 5 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT6_resnet18_cbam_kshot5.pth
```

#### **k_shot = 10**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT6/unseen --experiment MT6 --model resnet18_cbam --n_way 6 --k_shot 10 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT6_resnet18_cbam_kshot10.pth
```

#### **k_shot = 15**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT6/unseen --experiment MT6 --model resnet18_cbam --n_way 6 --k_shot 15 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT6_resnet18_cbam_kshot15.pth
```

#### **k_shot = 20**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT6/unseen --experiment MT6 --model resnet18_cbam --n_way 6 --k_shot 20 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT6_resnet18_cbam_kshot20.pth
```

---

## **Evaluation Commands for ResNet18**

### **MT-10 Experiment**

#### **k_shot = 1**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT10/unseen --experiment MT10 --model resnet18 --n_way 10 --k_shot 1 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT10_resnet18_kshot1.pth
```

#### **k_shot = 5**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT10/unseen --experiment MT10 --model resnet18 --n_way 10 --k_shot 5 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT10_resnet18_kshot5.pth
```

#### **k_shot = 10**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT10/unseen --experiment MT10 --model resnet18 --n_way 10 --k_shot 10 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT10_resnet18_kshot10.pth
```

#### **k_shot = 15**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT10/unseen --experiment MT10 --model resnet18 --n_way 10 --k_shot 15 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT10_resnet18_kshot15.pth
```

#### **k_shot = 20**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT10/unseen --experiment MT10 --model resnet18 --n_way 10 --k_shot 20 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT10_resnet18_kshot20.pth
```

### **MT-6 Experiment**

#### **k_shot = 1**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT6/unseen --experiment MT6 --model resnet18 --n_way 6 --k_shot 1 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT6_resnet18_kshot1.pth
```

#### **k_shot = 5**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT6/unseen --experiment MT6 --model resnet18 --n_way 6 --k_shot 5 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT6_resnet18_kshot5.pth
```

#### **k_shot = 10**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT6/unseen --experiment MT6 --model resnet18 --n_way 6 --k_shot 10 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT6_resnet18_kshot10.pth
```

#### **k_shot = 15**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT6/unseen --experiment MT6 --model resnet18 --n_way 6 --k_shot 15 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT6_resnet18_kshot15.pth
```

#### **k_shot = 20**

```bash
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT6/unseen --experiment MT6 --model resnet18 --n_way 6 --k_shot 20 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_MT6_resnet18_kshot20.pth
```
