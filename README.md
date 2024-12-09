# Google colab for running EZKL on model & data files:

https://colab.research.google.com/github/rainbowpuffpuff/ez_think2earn/blob/master/ZuThailand_EZKL_demo.ipynb 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rainbowpuffpuff/ez_think2earn/blob/master/ZuThailand_EZKL_demo.ipynb)

1. Click the "Open in Colab" button to access the notebook environment.
2. Run all cells by navigating to Runtime -> Run all.
3. Wait for approximately one minute as the notebook processes the fNIRS data. You’ll see a verified ezkl proof of a model trained to classify between two states: mental arithmetic activity or baseline observation.

Data source:
[tu-berlin](https://doc.ml.tu-berlin.de/hBCI/contactthanks.php)

NIRS data "NIRS_01-29". For the ZuThailand buildathon, only the first 9 participants are selected for processing. 

---
# Setup instructions for model training

1. git clone https://github.com/rainbowpuffpuff/ez_think2earn
2. cd fNIRSNET
3. python KFold_train.py -> wait for the training to be done and for files to be outputted in the "fNIRSNET/save/" folder
4. cd flask-backend
5. python app.py
6. In second terminal, cd ..
7. cd frontend
8. npm install
9. npm start
10. upload model.onnx from example pathing: ~/ez_think2earn/fnirs_ezkl/fNIRSNET/save/MA/KFold/1/1/
11. wait for printout
12. You successfully uploaded a brain computer interface model (https://github.com/wzhlearning/fNIRSNet), the script inputted data (https://doc.ml.tu-berlin.de/hBCI/contactthanks.php), and you created EZKL proofs that a private model was run on public data

Below there's results of the training done on the Mental Arithmetic dataset:

Explaining the confusion matrix:

The model correctly identifies 71% of non-mental arithmetic cases and 68% of mental arithmetic cases. However, it misclassifies 29% of non-mental arithmetic as mental arithmetic and 32% of mental arithmetic as non-mental arithmetic, indicating moderate performance with room for improvement.


![Confusion_Matrix_Overall](https://github.com/user-attachments/assets/0465d8f8-cfe2-4d29-83cf-3e9cffa5ba7d)
![Overall_Aggregated_Loss_Curve](https://github.com/user-attachments/assets/97d6af2b-ed3f-4109-a586-7efc6102dbd6)
![Overall_Aggregated_Accuracy_Curve](https://github.com/user-attachments/assets/81c0abd0-ea32-4548-b235-af9481174e53)
