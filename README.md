<div align="center">
  
# Classification of Human Movements in Time Series Using Long Short-Term Memory (LSTM)
  
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)[![License](https://img.shields.io/badge/license-MIT-green.svg)]

</div>
This project evaluates the performance of a Long Short-Term Memory (LSTM) architecture for human movement classification in the UCF50 dataset. 

## Introduction
Human movements consist of actions that cannot be properly classified by one image alone, but rather by a set of images in a specific sequence. In this context, the goal of this project is to address the problem of identifying movements by using multi-frame containers (videos) and creating a time-series neural network module. To achieve this goal,  a Long Short-Term Memory (LSTM) architecture was chosen due to its ability to retain information from previous steps. Furthermore, to evaluate the network, different frame inputs were tested from 15 to 120 frames.

Regarding the dataset, this study utilizes Realistic Action Recognition: UCF50 [[1](https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50/code)]. The main reasons for this choice are: the variety of human movement and consistency in usage worldwide.

## Architecture
To carry out this study, based on Bleed AI Academy’s Youtube video [[2](https://www.youtube.com/watch?v=QmtSkq3DYko)], the following architecture of LSTM was used:
<pre> 
                              ------------------------------------------------
                              |                 ConvLSTM2D                   |
                              ------------------------------------------------
                              |   Filters=4, Kernel=(3,3), Activation=Tanh   |
                              ------------------------------------------------
                                                   ↓
                              ------------------------------------------------
                              |                MaxPooling3D                  |
                              ------------------------------------------------
                              |       Padding=Same, Pool_Size=(1,2,2)        |
                              ------------------------------------------------
                                                   ↓
                              ------------------------------------------------
                              |        TimeDistributed + Dropout             |
                              ------------------------------------------------
                              |                Dropout=0.2                   |
                              ------------------------------------------------
                                                   ↓
                              ------------------------------------------------
                              |                 ConvLSTM2D                   |
                              ------------------------------------------------
                              |   Filters=14, Kernel=(3,3), Activation=Tanh  |
                              ------------------------------------------------
                                                   ↓
                              ------------------------------------------------
                              |                MaxPooling3D                  |
                              ------------------------------------------------
                              |       Padding=Same, Pool_Size=(1,2,2)        |
                              ------------------------------------------------
                                                   ↓
                              ------------------------------------------------
                              |        TimeDistributed + Dropout             |
                              ------------------------------------------------
                              |                Dropout=0.2                   |
                              ------------------------------------------------
                                                   ↓
                              ------------------------------------------------
                              |                 ConvLSTM2D                   |
                              ------------------------------------------------
                              |   Filters=16, Kernel=(3,3), Activation=Tanh  |
                              ------------------------------------------------
                                                   ↓
                              ------------------------------------------------
                              |                MaxPooling3D                  |
                              ------------------------------------------------
                              |       Padding=Same, Pool_Size=(1,2,2)        |
                              ------------------------------------------------
                                                   ↓
                              ------------------------------------------------
                              |                  Flatten                     |
                              ------------------------------------------------
                                                   ↓
                              ------------------------------------------------
                              |                  Dense                       |
                              ------------------------------------------------
                              |        6 classes, Activation=SoftMax         |
                              ------------------------------------------------
  </pre>

## Methods
Initially, to assess which configuration presents the best performance, it was decided to fix the number of classes to seven: WalkingWithDog, Skiing, Swing, Diving, Mixing, HorseRace, and HorseRiding. The classes are encoded with One-Hot Encoded Labels (no need for ordering among themselves).

After that establishment, the next step was to alter the quantity of collected frames from 15 to 120 frames. There, I trained each network in 5 epochs to expect an overall performance, and subsequently selected the more efficient ones for longer training (30 epochs).

For matters of evaluation, metrics such as loss, accuracy, recall, and precision were the backbone to appoint the best network for this context. Finally, the assessment was deemed successful.


## Results and Discussion
Initially, the metrics of mean loss, accuracy, precision, recall, and training time for each trained model will be discussed.

## Conclusion
The study demonstrated that LSTMs are a solution to human movement classification problems. Despite using a small and educational dataset, the trained model presented satisfactory results.  Furthermore, it is worth noting that, in terms of the UCF50 dataset, the overall best setting happens when 60 frames are captured from each video. 

##  Future Steps
It is worth noting that this repository is only a scratch of LSTM's potential to tackle problems concerning the identification of human movements. For the future, adding the capacity of continuous learning, designing an accessible user terminal to execute functions (such as training, creating a dataset, evaluating performance), and testing different architectures are possible implementations.

## References
[1] P. Ahmad, "Realistic Action Recognition - UCF50," Kaggle, 2022. [Online]. Available: [https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50](https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50).

[2] Bleed AI Academy, "Human Activity Recognition using TensorFlow (CNN + LSTM) | 2 Methods", YouTube, 2021. [Online]. Available: [https://www.youtube.com/watch?v=QmtSkq3DYko](https://www.youtube.com/watch?v=QmtSkq3DYko).
