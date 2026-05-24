# ==========================================
# REMOVE TENSORFLOW WARNINGS
# ==========================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# COMMAND LINE ARGUMENT
# ==========================================

ap = argparse.ArgumentParser()

ap.add_argument(
    "--mode",
    help="train/display"
)

mode = ap.parse_args().mode

# ==========================================
# FUNCTION TO PLOT ACCURACY & LOSS
# ==========================================

def plot_model_history(model_history):

    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    # ACCURACY GRAPH
    axs[0].plot(
        range(1, len(model_history.history['accuracy']) + 1),
        model_history.history['accuracy']
    )

    axs[0].plot(
        range(1, len(model_history.history['val_accuracy']) + 1),
        model_history.history['val_accuracy']
    )

    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')

    axs[0].legend(
        ['train', 'validation'],
        loc='best'
    )

    # LOSS GRAPH
    axs[1].plot(
        range(1, len(model_history.history['loss']) + 1),
        model_history.history['loss']
    )

    axs[1].plot(
        range(1, len(model_history.history['val_loss']) + 1),
        model_history.history['val_loss']
    )

    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')

    axs[1].legend(
        ['train', 'validation'],
        loc='best'
    )

    plt.savefig('training_plot.png')

    # IMPORTANT
    plt.close()

# ==========================================
# DATASET PATHS
# ==========================================

train_dir = 'fer2013_extracted/train'
val_dir = 'fer2013_extracted/test'

# ==========================================
# DATASET INFORMATION
# ==========================================

num_train = 28709
num_val = 7178

batch_size = 64

# BETTER ACCURACY
num_epoch = 50

# ==========================================
# IMAGE PREPROCESSING
# ==========================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True

)

val_datagen = ImageDataGenerator(

    rescale=1./255

)

# ==========================================
# LOAD TRAIN DATA
# ==========================================

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(48,48),

    batch_size=batch_size,

    color_mode="grayscale",

    class_mode='categorical'

)

# ==========================================
# LOAD VALIDATION DATA
# ==========================================

validation_generator = val_datagen.flow_from_directory(

    val_dir,

    target_size=(48,48),

    batch_size=batch_size,

    color_mode="grayscale",

    class_mode='categorical'

)

# ==========================================
# BUILD CNN MODEL
# ==========================================

model = Sequential()

# FIRST CNN BLOCK

model.add(

    Conv2D(

        32,

        kernel_size=(3,3),

        activation='relu',

        input_shape=(48,48,1)

    )

)

model.add(

    Conv2D(

        64,

        kernel_size=(3,3),

        activation='relu'

    )

)

model.add(

    MaxPooling2D(

        pool_size=(2,2)

    )

)

model.add(

    Dropout(0.25)

)

# SECOND CNN BLOCK

model.add(

    Conv2D(

        128,

        kernel_size=(3,3),

        activation='relu'

    )

)

model.add(

    MaxPooling2D(

        pool_size=(2,2)

    )

)

model.add(

    Conv2D(

        128,

        kernel_size=(3,3),

        activation='relu'

    )

)

model.add(

    MaxPooling2D(

        pool_size=(2,2)

    )

)

model.add(

    Dropout(0.25)

)

# FLATTEN LAYER

model.add(

    Flatten()

)

# DENSE LAYER

model.add(

    Dense(

        1024,

        activation='relu'

    )

)

model.add(

    Dropout(0.5)

)

# OUTPUT LAYER

model.add(

    Dense(

        7,

        activation='softmax'

    )

)

# ==========================================
# TRAIN MODEL
# ==========================================

if mode == "train":

    model.compile(

        loss='categorical_crossentropy',

        optimizer=Adam(learning_rate=0.0001),

        metrics=['accuracy']

    )

    print("\nTraining Started...\n")

    model_info = model.fit(

        train_generator,

        steps_per_epoch=num_train // batch_size,

        epochs=num_epoch,

        validation_data=validation_generator,

        validation_steps=num_val // batch_size

    )

    # SAVE TRAINING GRAPH

    plot_model_history(model_info)

    # SAVE FULL MODEL

    model.save('model.h5')

    print("\nModel trained and saved successfully!")

# ==========================================
# REAL-TIME EMOTION DETECTION
# ==========================================

elif mode == "display":

    # LOAD TRAINED MODEL

    model = load_model('model.h5')

    cv2.ocl.setUseOpenCL(False)

    # EMOTION LABELS

    emotion_dict = {

    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Surprised",
    6: "Sad"
}

    # LOAD FACE DETECTOR

    facecasc = cv2.CascadeClassifier(

        'haarcascade_frontalface_default.xml'

    )

    # CAMERA
    # 0 = laptop camera
    # 1 = external webcam

    cap = cv2.VideoCapture(0)

    print("\nPress 'q' to exit webcam\n")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY

        )

        faces = facecasc.detectMultiScale(

            gray,

            scaleFactor=1.3,

            minNeighbors=5

        )

        for (x, y, w, h) in faces:

            cv2.rectangle(

                frame,

                (x, y-50),

                (x+w, y+h+10),

                (255,0,0),

                2

            )

            roi_gray = gray[

                y:y+h,

                x:x+w

            ]

            cropped_img = np.expand_dims(

                np.expand_dims(

                    cv2.resize(

                        roi_gray,

                        (48,48)

                    ),

                    -1

                ),

                0

            )

            # IMPORTANT NORMALIZATION

            cropped_img = cropped_img / 255.0

            prediction = model.predict(

                cropped_img,

                verbose=0

            )

            maxindex = int(

                np.argmax(prediction)

            )

            cv2.putText(

                frame,

                emotion_dict[maxindex],

                (x+20, y-60),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (255,255,255),

                2,

                cv2.LINE_AA

            )

        cv2.imshow(

            'Emotion Detection',

            cv2.resize(

                frame,

                (1200,860)

            )

        )

        # PRESS q TO EXIT

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

    cap.release()

    cv2.destroyAllWindows()

# ==========================================
# INVALID MODE
# ==========================================

else:

    print("\nInvalid mode!")

    print("Use:")

    print("python emotion.py --mode train")

    print("OR")

    print("python emotion.py --mode display")