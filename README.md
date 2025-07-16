# Classification of Human Movements in Time Series Using Long Short-Term Memory (LSTM)
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
Initially, to assess which configuration presents the best performance, it was decided to fix the number of classes to seven: WalkingWithDog, Skiing, Swing, Diving, Mixing, HorseRace, and HorseRiding. The classes are encoded with One-Hot Encoded Labels (no need for ordering among themselves). After that establishment, the next step was to alter the quantity of collected frames from 15 to 120 frames. For matters of evaluation, metrics such as loss, accuracy, recall, and precision were the backbone to appoint the best network for this context. Finally, the assessment was deemed successful.

## Results and Discussion
...

## Conclusion
...

##  Future Steps
It is worth noting that this repository is only a scratch of LSTM's potential to tackle problems concerning the identification of human movements. For the future, adding the capacity of continuous learning, designing an accessible user terminal to execute functions (such as training, creating a dataset, evaluating performance), and testing different architectures are possible implementations.

## References
[1] [1] P. Ahmad, "Realistic Action Recognition - UCF50," Kaggle, 2022. [Online]. Available: [https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50](https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50)

[2] Bleed AI Academy, "Human Activity Recognition using TensorFlow (CNN + LSTM) | 2 Methods", YouTube, 2021. [Online]. Available: [https://www.youtube.com/watch?v=QmtSkq3DYko](https://www.youtube.com/watch?v=QmtSkq3DYko).
