setup instructions 
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
