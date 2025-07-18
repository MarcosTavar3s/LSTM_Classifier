<div align="center">
  
# Classification of Human Movements in Time Series Using Long Short-Term Memory (LSTM)
  
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/MarcosTavar3s/LSTM_Classifier/blob/main/LICENSE) [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
</div>

---

## 📋 Description 
This project evaluates the performance of a Long Short-Term Memory (LSTM) architecture for human movement classification in the UCF50 dataset. 

---

## 📜 Summary
- [1. Introduction](#1-introduction)
- [2. Architecture](#2-architecture)
- [3. Methods](#3-methods)
- [4. Results and Discussion](#4-results-and-discussion)
- [5. Conclusion](#5-conclusion)
- [6. Future Steps](#6-future-steps)
- [7. Running Locally](#7-running-locally) 
- [8. Developer Team](#8-developer-team)
- [9. References](#9-references)
- [10. License](#10-license)

---
## 📂 Project Tree
```
LSTM_Classifier/
├── code/
|   ├── examples # videos to test the network
│   ├── module.py
│   ├── main.py
|   ├── requirements.txt
├── statistics/ # statistic results
├── README.md
└── LICENSE
```
---

## 1. Introduction
Human movements consist of actions that cannot be properly classified by one image alone, but rather by a set of images in a specific sequence. In this context, the goal of this project is to address the problem of identifying movements by using multi-frame containers (videos) and creating a time-series neural network module. To achieve this goal,  a Long Short-Term Memory (LSTM) architecture was chosen due to its ability to retain information from previous steps. Furthermore, to evaluate the network, different frame inputs were tested from 15 to 120 frames.

Regarding the dataset, this study utilizes Realistic Action Recognition: UCF50 [[1](https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50/code)]. The main reasons for this choice are: the variety of human movement and consistency in usage worldwide.
<br>

<div align="center">

| Diving | HorseRace | Mixing |
| :-----: | :-----: | :-----: |
| <img src="code/examples/diving_gif.gif" alt="Diving gif" width="200"/> | <img src="code/examples/v_HorseRace_g11_c07.gif" alt="HorseRace gif" width="200"/> | <img src="code/examples/v_Mixing_g18_c02.gif" alt="Mixing gif" width="200"/> 

  _Example of Diving, HorseRace, and Mixing classes in UCF50 dataset_
  
</div>

---

## 2. Architecture
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


---

## 3. Methods
Initially, to assess which configuration presents the best performance, it was decided to fix the number of classes to seven: WalkingWithDog, Skiing, Swing, Diving, Mixing, HorseRace, and HorseRiding. The classes are encoded with One-Hot Encoded Labels (no need for ordering among themselves).

<div align="center">

| Skiing | HorseRiding | Swing |  WalkingWithDog |
| :-----: | :-----: | :-----: | :-----: |
| <img src="code/examples/v_Skiing_g24_c01.gif" alt="Skiing gif" width="150"/> | <img src="code/examples/v_HorseRiding_g16_c02.gif" alt="HorseRiding gif" width="150"/> | <img src="code/examples/v_Swing_g11_c02.gif" alt=" Swing gif" width="150"/> | <img src="code/examples/v_WalkingWithDog_g19_c05.gif" alt=" WalkingWithDog gif" width="150"/> |

_Example of Skiing, HorseRiding, Swing and WalkingWithDog classes in UCF50 dataset_

</div>
  
After that establishment, the next step was to alter the quantity of collected frames from 15 to 120 frames. There, I trained each network in 5 epochs to expect an overall performance, and subsequently selected the more efficient ones for longer training (30 epochs).

For matters of evaluation, metrics such as loss, accuracy, recall, and precision were the backbone to appoint the best network for this context. Finally, the assessment was deemed successful.

---

## 4. Results and Discussion
Initially, the metrics of mean loss, accuracy, precision, recall, and training time for each trained model will be discussed.

---

## 5. Conclusion
The study demonstrated that LSTMs are a solution to human movement classification problems. Despite using a small and educational dataset, the trained model presented satisfactory results.  Furthermore, it is worth noting that, in terms of the UCF50 dataset, the overall best setting happens when 60 frames are captured from each video. 

---

##  6. Future Steps
It is worth noting that this repository is only a scratch of LSTM's potential to tackle problems concerning the identification of human movements. For the future, adding the capacity of continuous learning, designing an accessible user terminal to execute functions (such as training, creating a dataset, evaluating performance), and testing different architectures are possible implementations.

---
## 7. Running Locally
📥 Clone the repository:
```bash
git clone https://https://github.com/MarcosTavar3s/LSTM_Classifier.git
cd code
```

🐍 Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate # Linux or MacOS
venv/Scripts/activate # Windows
```

📦 Install the dependencies:

```bash
pip install -r requirements.txt
```

🚀Run the project:
```bash
python main.py
```
---

## 8. Developer Team
| [![](https://github.com/MarcosTavar3s.png?size=80)](https://github.com/MarcosTavar3s) | [![](https://github.com/heltonmaia.png?size=80)](https://github.com/heltonmaia) |
| :-----------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: |
|                           **Marcos Aurélio**<br> Researcher                           |                       **Helton Maia** <br> Academic Advisor                     |

---
## 9. References
[1] P. Ahmad, "Realistic Action Recognition - UCF50," Kaggle, 2022. [Online]. Available: [https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50](https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50).

[2] Bleed AI Academy, "Human Activity Recognition using TensorFlow (CNN + LSTM) | 2 Methods", YouTube, 2021. [Online]. Available: [https://www.youtube.com/watch?v=QmtSkq3DYko](https://www.youtube.com/watch?v=QmtSkq3DYko).

---

## 10. License
This project is licensed under the terms of the [GNU General Public License v3.0](https://github.com/MarcosTavar3s/LSTM_Classifier/blob/main/LICENSE).
