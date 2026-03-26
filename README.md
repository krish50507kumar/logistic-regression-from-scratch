# logistic-regression-from-scratch
# Logistic — Custom Logistic Regression from Scratch

A fully customizable **Logistic Regression implementation** built from scratch using NumPy.  
Supports **binary + multiclass classification**, multiple optimizers, regularization methods, and preprocessing utilities.

---

## 🚀 Features

- ✅ Binary & Multiclass Classification
- ✅ Optimizers:
  - Adam
  - Momentum
  - RMSProp
  - Lion 
- ✅ Regularization:
  - L1 (Lasso)
  - L2 (Ridge)
  - ElasticNet
- ✅ Gradient Strategies:
  - Batch Gradient Descent
  - Mini-batch Gradient Descent
  - Stochastic Gradient Descent (SGD)
- ✅ Learning Rate Scheduling:
  - Step decay
  - Time decay
  - Exponential decay
  - Cosine decay
- ✅ Built-in Preprocessing:
  - Missing value handling
  - Label encoding
  - Feature selection & dropping
  - Normalization
- ✅ Early Stopping (basic patience-based)
- ✅ Model Saving & Loading

---

## 📦 Installation

No external ML libraries required.

```bash
pip install numpy
```
## ⚙️ Parameters

- Parameter	Description	Default
- kind	"binary" or "multiclass"	"binary"
- reg	"L1", "L2", "ElasticNet", "None"	"L1"
- gradient	"batch", "minibatch", "sdc"	"minibatch"
- optimizer	"adam", "momentum", "rmsprop", "lion"	"adam"
- alpha	ElasticNet mixing ratio	0.5
- lamb	Regularization strength	0.5
- Lr	Learning rate	0.001
- batch_size	Batch size	64
- decay	LR decay strategy	"step"
- strategy	Missing value handling	"mean"

---

## 🧠 Usage

```Basic Example
model = Logistic_0(kind="binary", reg="L2", gradient="minibatch", Lr=0.01)

X_train, X_test, y_train, y_test = model.split(X, y, seed=42, shuffle=True)

model.train(X_train, y_train, X_test, y_test, epochs=500)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
```
---
```Full Pipeline Example
model = Logistic_0()

# Feature engineering
X = model.select_features(X, labels)
X = model.drop_features(X, labels)

# Encoding + cleaning
D = model.label_encoder(X+y,labels)
X,y = D[:,:-1],D[:,-1]
X = model.clean(X)

# Normalization
X = model.normalization(X)

# Split
X_train, X_test, y_train, y_test = model.split(X, y, seed=42, shuffle=True)

# Train
model.train(X_train, y_train, X_test, y_test, epochs=500)

# Evaluate
print(model.score(X_test, y_test))

# Save & Load
model.save("model.pkl")
model = Logistic.load("model.pkl")
```

---

## 📊 Key Methods

- Training
  - train(X, y, Xt, yt, epochs)
  - Supports early stopping + LR decay
- Prediction
  - predict(X) → class labels
  - score(X, y) → accuracy
- Preprocessing
  - clean(X) → handle NaNs
  - label_encoder(X, cols)
  - normalization(X)
  - split(X, y, seed, shuffle)
- Feature Engineering
  - select_features(X, cols)
  - drop_features(X, cols)
- Model Utilities
  - summary() → prints configuration
  - save(path) / load(path)

---

## 🧩 Internal Design (High-Level)

- Forward pass:
  - Binary → Sigmoid
  - Multiclass → Softmax
- Loss:
  - Binary Cross Entropy
  - Categorical Cross Entropy
- Backprop:
  - Vectorized gradients
- Regularization added to:
  - Loss
  - Gradient update

---

## ⚠️ Limitations 

- Not production-ready (no GPU, no batching optimizations)
- No advanced metrics (precision, recall, F1)
- No cross-validation
- Label encoding is basic (no unseen category handling)
- Early stopping is simplistic
