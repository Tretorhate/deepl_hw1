# Терең оқыту - 1-тапсырма (Жеңілдетілген нұсқа)
## Сипаттама

Бұл жеңілдетілген нұсқа - эпохалар саны азайтылған, тез орындалу үшін.

| Есеп              | Түпнұсқа   | Жеңіл     |
| ----------------- | ---------- | --------- |
| MLP оқыту         | 20 эпоха   | 3 эпоха   |
| XOR оқыту         | 1000 эпоха | 200 эпоха |
| Оптимизаторлар    | 20 эпоха   | 3 эпоха   |
| Градиент талдау   | 10 эпоха   | 2 эпоха   |
| CNN оқыту         | 10 эпоха   | 2 эпоха   |
| Transfer Learning | 10 эпоха   | 2 эпоха   |

## Орнату

### 1. Қажетті кітапханалар

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

### 2. GPU (міндетті емес)

CUDA бар болса, автоматты түрде GPU қолданылады.

## Іске қосу

### Барлығын бірден орындау

```bash
cd assg_1_lighweight
python run_all.py
```

### Жеке есептерді орындау

```bash
# 1-Есеп: MLP
python problem1_mlp.py

# 2-Есеп: Оптимизация
python problem2_optimization.py

# 3-Есеп: CNN
python problem3_cnn.py

# Бонус: Transfer Learning
python bonus_transfer_learning.py
```

## Файлдар

```
assg_1_lighweight/
├── problem1_mlp.py          # MLP және XOR
├── problem2_optimization.py  # Оптимизаторлар, градиенттер
├── problem3_cnn.py          # CNN CIFAR-10
├── bonus_transfer_learning.py # ResNet18 transfer
├── run_all.py               # Барлығын орындау
├── report.tex               # LaTeX есеп
├── README.md                # Осы файл
└── results/                 # Нәтижелер (графиктер)
    ├── part1b_loss_comparison.png
    ├── part1c_xor_boundary.png
    ├── part1c_xor_loss.png
    ├── part2a_optimizer_comparison.png
    ├── part2b_gradient_analysis.png
    ├── part2c_regularization.png
    ├── part3a_training_curves.png
    ├── part3a_confusion_matrix.png
    ├── part3b_architecture_comparison.png
    ├── part3c_filters.png
    ├── part3c_activations.png
    └── bonus_transfer_learning.png
```

## Есептер

### 1-Есеп: MLP

- **A**: MNIST-те MLP оқыту (784→128→64→10)
- **B**: Активация функцияларын салыстыру (ReLU, Sigmoid, Tanh)
- **C**: XOR есебін шешу

### 2-Есеп: Оптимизация

- **A**: Оптимизаторларды салыстыру (SGD, Momentum, RMSprop, Adam)
- **B**: Градиент жоғалу мәселесін талдау
- **C**: Регуляризация әдістері (L2, Dropout)

### 3-Есеп: CNN

- **A**: CIFAR-10-те CNN оқыту
- **B**: Архитектура эксперименттері
- **C**: Фильтрлер мен активацияларды визуализациялау

### Бонус: Transfer Learning

- ImageNet-те оқытылған ResNet18-ді CIFAR-10-ге бейімдеу

## LaTeX есебін құрастыру

```bash
pdflatex report.tex
```

## Күтілетін уақыт

- CPU: ~5-10 минут
- GPU: ~2-5 минут
