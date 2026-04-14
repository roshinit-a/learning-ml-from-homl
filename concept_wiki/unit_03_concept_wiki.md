## 📘 Chapter 3: Model Evaluation

---

### 1. Cross Validation

Cross-validation is a technique to evaluate **how well a model generalizes** to unseen data.

In `sklearn.model_selection.cross_val_score()`, the parameter `cv` represents the number of folds (default = 5). 

- **The `scoring` parameter:** This tells Scikit-Learn exactly which metric to compute on each fold (e.g., `scoring="accuracy"`, `scoring="f1"`).

#### 🔁 Working (K-Fold Example: cv = 3)

The dataset is split into 3 equal parts: A, B, C.

- Iteration 1 → Train on (A + B), Test on C  
- Iteration 2 → Train on (B + C), Test on A  
- Iteration 3 → Train on (A + C), Test on B  

Each data point is used for **training and testing exactly once**.

- **What to do with the scores:** You get an array of `k` scores back. You never just look at one fold! Always take the **mean** (your overall performance estimate) and the **standard deviation** (how stable/consistent your model is).
- **Stratified Sampling:** Cross-validation uses `StratifiedKFold` for classification. This preserves the exact class ratio in each fold (e.g., ensuring every fold has exactly 10% '5s'), preventing bad luck random splits!

---

### 2. Confusion Matrix

A confusion matrix summarizes **how well a classification model performs** by comparing actual vs predicted labels.

In `sklearn.metrics.confusion_matrix()`:

    array([[true negative, false positive],
           [false negative, true positive]])

#### 📊 Structure

- Rows → Actual (True labels)  
- Columns → Predicted (Model output)

|                       | Predicted Negative | Predicted Positive |
|-----------------------|-------------------|-------------------|
| **Actual Negative** | True Negative (TN)| False Positive (FP)|
| **Actual Positive** | False Negative (FN)| True Positive (TP)|

---

### 🔍 Definitions

- **True Negative (TN):** Correctly predicted negative  
- **False Positive (FP):** Incorrectly predicted positive (Type I Error)  
- **False Negative (FN):** Incorrectly predicted negative (Type II Error)  
- **True Positive (TP):** Correctly predicted positive  

---

### 3. Precision & 4. Recall (The Trade-off)


These two metrics are not independent; they are locked in a constant tug-of-war. 

- **Precision:** If my model screams "This is a 5!", how often is it actually right?
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

- **Recall:** Out of all the real 5s in the dataset, how many did my model actually find?
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

🚨 **The Trade-off:** Increasing precision usually reduces recall, and vice versa. 
Raising the decision threshold makes the model more selective. It gains **higher precision** (fewer false alarms), but it misses more actual positives, resulting in **lower recall**.

---

### 5. F1 Score

F1 Score is the **harmonic mean of precision and recall**. It gives you a single metric to evaluate your model's balance.

#### 📌 Why the Harmonic Mean instead of Arithmetic Mean?
The harmonic mean severely **punishes imbalance**. 
- If Precision = 1.0 and Recall = 0.0, a standard arithmetic mean gives `0.5` (which looks misleadingly decent). 
- The harmonic mean gives `0` (which correctly shows the model is terrible). It forces the model to have *both* a good precision and a good recall to score well.

#### 📌 Harmonic Mean (General Formula)

$$
H = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}
$$

#### 📌 F1 Formula

$$
F_1 = \frac{2}{\frac{1}{\text{precision}} + \frac{1}{\text{recall}}}
= 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$