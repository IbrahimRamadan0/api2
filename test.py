from os import EX_CANTCREAT
import cv2
import tensorflow as tf
import numpy as np
import sys
from keras.models import load_model

if (len(sys.argv) < 2):
    print("how can use this script::-->python project.py image_path")
else:
    # LABELS = ["a1","a2","a3","a4","a5","abo_elhgag","kapsh" , "masla" , "not_recogmize" , "status1" , "status2" , "status3" , "status4","status5" , "status6" , "status7" , "status8", "status9","wall1" , "wall2"]
    # LABELS = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","Abou_al_Haggag_Mosque","Akhenaten","Tutankhamun","amnhoutb_iii","hatshepsut","nkhtnbo_i","ramses_ii"]      #LABELS = ["17","2","Abou_al_Haggag_Mosque","Akhenaten","Tutankhamun","amnhoutb_iii","hatshepsut","nkhtnbo_i","ramses_ii"]
    # LABELS = ["Abou_al_Haggag_Mosque","Akhenaten","Tutankhamun","amnhoutb_iii","hatshepsut","lake_19","masla_21","nkhtnbo_i","ramses_ii","wall_1","wall_10","wall_11","wall_12","wall_13","wall_14","wall_15","wall_16","wall_17","wall_18","wall_2","wall_3","wall_4","wall_5","wall_6","wall_7","wall_8","wall_9"]
    LABELS = ["Abou_al_Haggag_Mosque", "Akhenaten", "Tutankhamun", "amnhoutb_iii", "hatshepsut", "lake_19", "masla_21",
              "nkhtnbo_i", "ramses_ii", "wall_1", "wall_10", "wall_11", "wall_12", "wall_13", "wall_14", "wall_15",
              "wall_16", "wall_17", "wall_18", "wall_2", "wall_3", "wall_4", "wall_5", "wall_6", "wall_7", "wall_8",
              "wall_9", "wall_19"]


    def prepare(filepath):
        IMG_SIZE = 250
        img_array = cv2.imread(filepath)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


    x = tf.keras.Input(shape=(250, 250, 3))
    y = tf.keras.layers.Dense(16, activation='softmax')(x)
    model = tf.keras.Model(x, y)
    model = load_model('/content/drive/MyDrive/v8/model2.h5')
    try:
        prediction = model.predict([prepare(sys.argv[1])])

        a = prediction
        i, j = np.unravel_index(a.argmax(), a.shape)
        a[i, j]
        # print(prediction)
        if (prediction[0][j] >= .6):
            print(LABELS[j])
        else:
            print("not_recognize ya Eslam :)")

    except:
        print("incorrect image path")