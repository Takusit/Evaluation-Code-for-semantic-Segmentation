# Evaluation-Code-for-semantic-Segmentation
This repository contains the pytorch code of semantic segmentation evaluation, which can be used to calculate commonly scores.
# How to start
1: Find lines 160 and 161 of the code
2: These two paths correspond to your predicted result and the real label, modify it to your local path！
3: Run!
# What will you get
1. Accuracy = (TP + TN) / (TP + TN + FP + TN)
2. Precision = TP / (TP + FP)
3. meanAccuracy
4. MIoU
5. Recall = TP / (TP + FN)
6. F1_score = (2 * Precision * Recall) / (Precision + Recall)
7. Dice_score
# The End
If my code helps you, please give me some attention and stars to encourage, thank you!
