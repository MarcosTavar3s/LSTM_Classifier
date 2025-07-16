# Classification of Human Movements in Time Series Using Long Short-Term Memory (LSTM)
This project evaluates the performance of a Long Short-Term Memory (LSTM) architecture for human movement classification in the UCF50 dataset. 

## Introduction
Human movements consist of actions that cannot be properly classified by one image alone, but rather by a set of images in a specific sequence. In this context, the goal of this project is to address the problem of identifying movements by using multi-frame containers (videos) and creating a time-series neural network module. To achieve this goal,  a Long Short-Term Memory (LSTM) architecture was chosen due to its ability to retain information from previous steps. Furthermore, to evaluate the network, different frame inputs were tested from 15 to 120 frames.

Regarding the dataset, this study utilizes Realistic Action Recognition: UCF50 [[1](https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50/code)]. The main reasons for this choice are: the variety of human movement and consistency in usage worldwide.

## Architecture
To carry out this study, based on Bleed AI Academy’s Youtube video [2], the following architecture of LSTM was used:
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
                                |          6 classes, Activation=SoftMax       |
                                ------------------------------------------------
  </pre>

## Methods
...

## Results and Discussion
...

## Conclusion
...

## References
[Num] Bleed AI Academy, "Human Activity Recognition using TensorFlow (CNN + LSTM) | 2 Methods", YouTube, 2021. [Online]. Available: [https://www.youtube.com/watch?v=QmtSkq3DYko](https://www.youtube.com/watch?v=QmtSkq3DYko).
