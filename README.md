# 🖼️ Image Compression using Singular Value Decomposition (SVD)

## 📚 Overview

This project is part of my **Machine Learning learning journey**.  
After learning **Singular Value Decomposition (SVD)** from linear algebra, I built an **image compression system from scratch** using **Python and NumPy** — **without using built-in image compression libraries**.

The goal was to deeply understand how **matrix decomposition** and **low-rank approximation** can be applied to real-world data such as images.

The system compresses images by **retaining only the most important singular values** and reconstructing the image from them.

---

## 🧠 Key Concepts Demonstrated

| Concept                                | Explanation                                                                 |
| ------------------------------------- | --------------------------------------------------------------------------- |
| **Singular Value Decomposition (SVD)** | Decomposes an image matrix into rotation and scaling components.            |
| **Low-Rank Approximation**             | Approximates the image using fewer singular values.                          |
| **Dimensionality Reduction**           | Reduces data size while preserving important visual features.               |
| **Matrix Reconstruction**              | Rebuilds the image using selected singular values.                           |
| **Compression–Quality Trade-off**      | Shows how image quality improves as more singular values are retained.      |

By implementing SVD manually, this project reinforces **linear algebra fundamentals + machine learning intuition**.

---

## 🗂️ Project Structure

```markdown
Image-Compression-Using-SVD/
├── Images/
│   └── skull.jpg
│
├── Outputs/
│   ├── compressed_k=1.jpg
│   ├── compressed_k=10.jpg
│   ├── compressed_k=50.jpg
│   └── compressed_k=100.jpg
│
├── image_compression(svd).py
└── README.md
```
## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```bash

git clone https://github.com/DolaSreecharan/Image-Compression-Using-SVD.git
cd Image-Compression-Using-SVD
```
### 2️⃣ Install required libraries

```bash

pip install numpy pillow
```
### 3️⃣ Run the script

```bash

python image_compression(svd).py
```
### 4️⃣ Enter number of singular values (k) when prompted

- Enter the number of singular values to keep: 50

## 📊 How the System Works

- Input Image is Converted into Matrix form using Numpy
- SVD applied
- For Gray Scale it is Directly Apply
- for R G B colors take 3 Channels and apply SVD for Each one
- Singular values are sorted in Descending Order
- Take top K singular Values
- reconstruct matrix
- Formula = Uk @ Σk @ V.Tk

## 🙋 About This Project

I am building small but meaningful projects as I progress through machine learning concepts.
This project represents my milestone in understanding how mathematical decompositions can be applied to image compression.

- Learning → Applying → Documenting → Showcasing

## ⭐ Support

If you find this project helpful, consider giving the repository a **star** — it supports and motivates my ML journey.

GitHub Profile: [https://github.com/DolaSreecharan](https://github.com/DolaSreecharan)





