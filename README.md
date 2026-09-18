# Machine Learning Models and Dataset for HMD Interaction Methods

This repository provides pre-trained models and a dataset trained using right-hand hand data acquired with the hand tracker of an XR headset.
Model 1 performs hand pose classification, while Model 2 detects touch motions.
The hand data is based on [OpenXR](https://learn.microsoft.com/ja-jp/windows/mixed-reality/develop/native/openxr) and consists of 26 joint points.

## Model 1

### Dataset

This model is designed for hand pose classification. The modelData1 dataset consists of right-hand hand data and is classified into the following two classes: "hand poses with only the index finger extended" and "other hand poses."
The dataset consists of hand data from eight participants.
The hand configurations and the number of samples per participant for each class are shown below.

#### Pose 1: Hand pose with only the index finger extended

This dataset consists of 1,000 samples of the hand pose shown in Figure 1.

![Pose1](https://github.com/user-attachments/assets/8a370059-32ef-40c9-893c-f6b20a622bae)
Figure 1

#### Pose 2: Other hand poses

This dataset consists of a total of 1,000 samples of the four types of hand poses shown in Figure 2.

![Pose2](https://github.com/user-attachments/assets/8a370059-32ef-40c9-893c-f6b20a622bae)
Figure 2

### Program

The program for training and evaluating the machine learning model using the dataset is available [here](model1.py).

### Pre-trained Model

The [ONNX-format pre-trained model](model1.onnx) provided in this repository classifies hand poses based on right-hand hand data.

## Model 2

### Dataset
This model is designed to detect touch motions.
A touch motion refers to a movement in which the right hand is thrust forward, as shown in Figure 3.
The modelData2 dataset consists of right-hand hand data and the position and rotation of the HMD, and is classified into two classes: "touch motion" and "other motions."
Other motions refer to movements in which the right hand is moved up, down, left, or right, or kept nearly stationary.
The dataset consists of hand data from eight participants.

![Pose3](https://github.com/user-attachments/assets/8a370059-32ef-40c9-893c-f6b20a622bae)
Figure 3

### Program

The program for training and evaluating the machine learning model using the dataset is available [here](model2.py).

### Pre-trained Model

The [ONNX-format pre-trained model](model2.onnx) provided in this repository detects touch motions based on right-hand hand data and the position and rotation of the XR headset.
