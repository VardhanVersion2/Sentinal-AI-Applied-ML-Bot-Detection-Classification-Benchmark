# Sentinal AI — Model Benchmark Leaderboard

| Model                          |   CV F1 (Mean) |   CV ROC-AUC |   Test Accuracy |   Test Precision |   Test Recall |   Test F1-Score |   Test ROC-AUC |   Test PR-AUC |   Train Time (s) |   Inference Latency (ms/sample) |
|:-------------------------------|---------------:|-------------:|----------------:|-----------------:|--------------:|----------------:|---------------:|--------------:|-----------------:|--------------------------------:|
| Logistic Regression (Baseline) |       0.719717 |     0.860958 |        0.786458 |         0.641805 |      0.806841 |        0.714922 |       0.863007 |      0.744483 |             0.05 |                          0.0011 |
| Random Forest                  |       0.818272 |     0.939174 |        0.885016 |         0.85614  |      0.785513 |        0.819307 |       0.940888 |      0.9115   |             3    |                          0.0228 |
| XGBoost Classifier             |       0.817438 |     0.941487 |        0.876068 |         0.798619 |      0.837827 |        0.817753 |       0.941763 |      0.91248  |             0.7  |                          0.0051 |

*Evaluation Protocol: Stratified 5-Fold CV + Isolated 20% Held-out Test Set (N=7,488)*
