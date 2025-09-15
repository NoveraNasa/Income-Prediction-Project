# Income Prediction Project

<!-- RESULTS:START -->

### 📊 Results (auto-generated)

**Best model by F1:** `LogisticRegression` (F1 = 0.665)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Best Params | CV F1 |
|---|---:|---:|---:|---:|---:|---|---:|
| LogisticRegression | 0.857 | 0.776 | 0.582 | 0.665 | 0.905 | `C`=1 | 0.651 |
| RandomForest | 0.849 | 0.746 | 0.578 | 0.651 | 0.904 | `max_depth`=20, `min_samples_leaf`=1, `n_estimators`=400 | 0.655 |

#### 🖼 Diagnostic Plots
- **LogisticRegression**  
  ![Confusion LogisticRegression](outputs/confusion_LogisticRegression.png)  
  ![ROC LogisticRegression](outputs/roc_LogisticRegression.png)
- **RandomForest**  
  ![Confusion RandomForest](outputs/confusion_RandomForest.png)  
  ![ROC RandomForest](outputs/roc_RandomForest.png)


<!-- RESULTS:END -->
