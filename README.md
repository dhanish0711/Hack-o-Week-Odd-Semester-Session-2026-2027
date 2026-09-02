# Hack-o-Week Odd Semester Session 2026-2027

Welcome to the **Hack-o-Week** repository for the Odd Semester Session. This repository contains weekly data engineering, web applications, machine learning foundations, and analytics projects developed week-by-week.

---

## 🛠️ Global Technology Stack

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/NumPy-Linear_Algebra-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Plots-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-000000?style=for-the-badge&logo=flask&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-07405E?style=for-the-badge&logo=sqlite&logoColor=white)
![Math](https://img.shields.io/badge/Mathematics-Calculus_&_Autograd-555555?style=for-the-badge&logo=latex&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-Structure-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-Styling-1572B6?style=for-the-badge&logo=css3&logoColor=white)

---

## 🗺️ Weekly Curriculum Roadmap

```mermaid
flowchart LR
    W1["📅 Week 1-2<br><b>Student Registry Portal</b><br>• Server-rendered CRUD<br>• SQLite Database<br>• REST API Seed Integration"]
    --> W3["📅 Week 3-4<br><b>E-Commerce Analytics Engine</b><br>• Python OOP & Comprehensions<br>• NumPy Array Broadcasting<br>• Pandas Multi-Table GroupBy"]
    --> W5["📅 Week 5-6<br><b>Linear Algebra & Autograd</b><br>• Vector Norms & Dot Products<br>• Covariance & Eigendecomposition<br>• Reverse-Mode AutoDiff DAG"]
    --> W7["📅 Week 7-8<br><b>Classical Machine Learning</b><br>• Linear, Poly, Ridge, Lasso<br>• Logistic Regression & KNN<br>• Car Valuation & Deal Advisor"]
```

---

## 📂 Repository Index & Subfolders

### 1. [week 1-2/](file:///e:/Hack-o-Week-Odd-Semester-Session-2026-2027/week%201-2) — Student Registry & Database CRUD Portal
- **Description**: Server-rendered Flask web application managing academic roll lists and student profiles with zero client-side JavaScript.
- **Key Features**: Live search filtering, dynamic admission date formatting, SQLite database persistence, and RandomUser API seed integration.

### 2. [week 3-4/](file:///e:/Hack-o-Week-Odd-Semester-Session-2026-2027/week%203-4) — Global E-Commerce Data Analytics & Visualization Engine
- **Description**: Python data engineering pipeline analyzing 397,000+ transaction records across 4,300+ customers using real relational datasets.
- **Key Features**:
  - **Python OOP & Comprehensions**: Modular architecture (`DataLoader`, `DataCleaner`, `SalesAnalyticsEngine`, `DataVisualizer`) with list/dict comprehensions.
  - **NumPy Array Vectorization & Broadcasting**: Array math (`np.multiply`), Z-score normalization `(price - mean) / std`, and RFM matrix dot products (`np.dot`).
  - **Pandas DataFrames & GroupBy**: 3-table SQL-style inner joins and multi-level GroupBy statistical aggregations.
  - **Data Visualizations**: High-resolution Seaborn and Matplotlib exports (Heatmaps, Violin Plots, Bar Charts, Line Dashboards).

### 3. [week 5-6/](file:///e:/Hack-o-Week-Odd-Semester-Session-2026-2027/week%205-6) — Linear Algebra & Calculus Engine for Machine Learning
- **Description**: Mathematics, linear algebra, and automatic differentiation (Autograd) system built from scratch in Python applied to the Hotel Booking Demand & Cancellation dataset (119,390 records).
- **Key Features**:
  - **Linear Algebra**: Feature vector $L_2$ norms, dot product cosine similarity matrices, covariance matrix ($\mathbf{\Sigma} = \frac{1}{N-1}\mathbf{X}^T\mathbf{X}$), spectral eigendecomposition ($\mathbf{\Sigma}\mathbf{v} = \lambda\mathbf{v}$), power iteration, and 2D PCA projection.
  - **Calculus & Autograd DAG**: Custom scalar `Value` computational graph, forward activations, analytical reverse-mode automatic differentiation using the Multivariate Chain Rule, and central finite difference gradient checking ($< 10^{-10}$ error).
  - **From-Scratch Neural Network**: Multi-Layer Perceptron (MLP) trained via pure backpropagation and SGD for cancellation prediction with loss curves and decision boundary plots.
  - **Diagnostic Visualizations**: High-resolution exports (Vector similarity heatmap, Covariance scree plot, PCA projection scatter, Gradient flow validation, Decision boundary surface).

### 4. [week 7-8/](file:///e:/Hack-o-Week-Odd-Semester-Session-2026-2027/week%207-8) — Classical Machine Learning Suite: Regression & Classification
- **Description**: Comprehensive machine learning suite covering classical Regression and Classification algorithms applied to the Used Car Resale Market Dataset (6,000 records).
- **Key Features**:
  - **Regression Suite**: Linear Regression (OLS & Normal Equation), Polynomial Regression (Degree 2, capturing non-linear depreciation and bias-variance tradeoff, $R^2 = 0.9318$), Ridge Regression ($L_2$ shrinkage), and Lasso Regression ($L_1$ coordinate descent feature sparsity).
  - **Classification Suite**: Logistic Regression (Binary Cross-Entropy, 79.92% accuracy, 0.835 ROC-AUC) and K-Nearest Neighbors (KNN with $k$-hyperparameter tuning).
  - **Real-World Decision Support**: Interactive Car Valuation & Deal Advisor evaluating specific vehicle buyer profiles and scoring deal quality.
  - **Diagnostic Visualizations**: 5 high-resolution figures (Depreciation curves, Ridge vs Lasso regularization paths, ROC/Confusion Matrix, KNN decision surface, Benchmark leaderboard).

---

Made by [Dhanish Ladwani](https://github.com/dhanish0711/)
