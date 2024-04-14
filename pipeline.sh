#!/bin/sh
echo "<< Start pipeline.sh >>"
echo " "
python data_creation.py
python model_preprocessing.py train/df_train_1.csv
python model_preprocessing.py train/df_train_2.csv
python model_preprocessing.py test/df_test_1.csv
python model_preprocessing.py test/df_test_2.csv
