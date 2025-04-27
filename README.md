# NdLinear-Enhanced CNN for Image Classification (SVHN Dataset)

In this project, I built a full deep learning pipeline for image classification using a CNN enhanced with NdLinear layers.
n this project, I built a deep learning pipeline for image classification using a CNN enhanced with NdLinear layers. The model is trained and evaluated on the SVHN (Street View House Numbers) dataset — a challenging real-world digit recognition task.

Instead of flattening feature maps early (which loses spatial relationships), I integrated NdLinear — a powerful module that applies N-dimensional linear transformations. This allows the model to preserve spatial structure across multiple dimensions, leading to better generalization, fewer parameters, and higher performance compared to standard CNNs with traditional dense layers

# Key Features
- **NdLinear-based CNN architecture** (preserves multi-dimensional structure)
- **SVHN Dataset** (real-world house number images)
- **Training pipeline** with:
  - Data loaders (`get_data_loaders`)
  - Training and evaluation loop (`train_model`)
  - Learning rate scheduling
- **Fine-tuning support**:
  - Adjustable learning rate, epochs, batch size
  - Basic data augmentation (RandomCrop, HorizontalFlip)
- **Visualizations**:
  - Training loss and accuracy plots
  - Confusion matrix
  - Random sample predictions
- **Modular Code Structure** (`src/model.py`, `src/train.py`, `run.py`)
- **Lightweight and scalable** — designed for easy extension to larger datasets

---

## 📈 Results

- **Test Accuracy:** ~85% after 20 epochs
- **Training Loss:** Smooth and consistently decreasing
- **Performance:** Stable training without overfitting

NdLinear significantly improved model performance by preserving feature structure without increasing computational costs.

---

## 📚 About NdLinear

**NdLinear** is a novel layer that applies **multi-dimensional linear transformations** to tensor inputs.

Instead of flattening tensors into vectors, it applies linear mappings along each axis independently, preserving spatial relationships. This technique is inspired by tensor decomposition methods (like Tucker decomposition) and is highly efficient.

**Benefits:**
- Better feature learning
- Reduced parameters
- Plug-and-play replacement for standard `nn.Linear`

**Reference Paper:**  
[NdLinear Is All You Need for Representation Learning (arXiv:2503.17353v1)](https://arxiv.org/abs/2503.17353)

---

