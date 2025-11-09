# Deep Learning Projects with PyTorch

This repository is a collection of **Deep Learning application projects** using **PyTorch** I have learned through multiple courses to practice with deep learning models and PyTorch.

The main goal is to strengthen fundamental knowledge of Deep Learning, get familiar with PyTorch and apply into real-world use cases. The projects cover diverse input types: structured data, sequences, images, text using multiple deep learning models (CNN, RNN, LSTM, GRU, Transformers) to tackle a variety tasks such as classification, prediction,... 

Along the way, applying real-world projects, tinkering with tensors, dealing with preprocessing data and evaluating helps me to learn a lot. Check my [mini project experiment on Substack blog](https://quynhanhphuong.substack.com/s/one-more-experiment) for more detailed experiments.

---

## Deep Learning basic and its application

| **Id** | **Data Types** | **Tag** | **Title** | **Description** | **Dataset** | **Methods/Models** | **Metrics** | **Results** | **Note** |
|---|---|---|---|---|---|---|---|---|---|
|   1  |   Structured data  |   Binary Classification  |   Detecting Cybersecurity Threats  |   Classify whether a threat or not  |   [BETH dataset](https://www.kaggle.com/datasets/katehighnam/beth-dataset)  |   Neural Networks  |   Accuracy  |   0.9448  |   Done  |
|   2  |   Image  |   Image Classification  |   Clouds image classification  |   Classify cloud images into category  |   Cloud image  |   Convolutional Neural Network  |   Precision, Recall  |     |   Done  |
|   3  |   Image  |   Multiclass Image Classification  |   E-commerce Clothing classifier  |   Classify clothing images into category  |   Clothing image  |   Convolutional Neural Network  |   Precision, Recall  | 0.87  0.77 |   Done  |
|   4  |   Sequence data  |   Time series prediction  |   Predicting electricity consumption  |     |   Electricity Consumption  |   RNN, LSTM, GRU  |   MSE, RMSE  |  0.04 0.2   |   Done  |
|   5  |   Sequence data  |   Time series prediction  |   Predicting traffic volume  |     |   Traffic volume by hour  |   RNN, LSTM, GRU  |   MSE, RMSE  |  0.071 0.26   |   Done  |
|   6  |   Text  |   Text Multiclass classification  |   Customer service text multiclass classification  |   Classify CS ticket into 5 categories |   Customer Support ticket  |   CNN  |   Accuracy, Precision, Recall  |  0.7892  0.7931  0.7892   |   Done  |
|   7  |   Structured data  |   Regression  |   Concrete strength prediction  |   Predict concrete strength based on their attributes |   [Concrete Dataset](https://www.kaggle.com/datasets/zain280/concrete-data/data)  |   Multi Layer Perceptron  |   MSE, RMSE  |   136  11.68  |   Done  |
|   8  |   Text  |   Text classification  |   Sentiment Analysis using Fine-tuning BERT  |   Sentiment Analysis (Negative, Positive) |     |   MLP  |   MSE, RMSE  |   136  11.68  |   Sample done  |
|   9  |   Structured data  |   Regression  |   Road Incident Prediction  |   Predict Incident risk (from 0 to 1) based on characteristics of road |     |   Multi Layer Perceptron  |   MSE, RMSE  |     |   Done  |

## Advanced Deep Learning
| **Id** | **Tags** | **Topic** | **Data types** | **Applications** | **Dataset** | **Metric and Results** | **Status** | **Note** |
|---|---|---|---|---|---|---|---|---|
| 1 |  | Auto Encoder | Image | Image Compression |  |  | Done |  |
| 1.1 |  | Auto Encoder | Image | Denoise Image |[Kaggle Denoise Dataset](https://www.kaggle.com/code/jesucristo/super-resolution-demo-swin2sr-official) |  | Done |  |
| 2 |  | Inception |  | Image Classification |  |  | In progress |  |
| 3 |  | Sequence to Sequence |  | Time Series Prediction |  |  |  | machinetranslation, imagecaptioning, textsummarization, time-series prediction, code generation |
| 4 |  | Generative Adversarial Networks (GAN)s fundamentals |  |  |  |  |  |  |
| 5 |  | Graph Neural Networks |  |  | PubMed |  | Done |  |
| 6 | #computervison  | Vision Transformers |  |  |  |  |  |  |
| 7 |  | Semi-Supervised Learning |  | Text Classification with pseudo-labeling |  |  |  |  |


## Techniques for efficient AI model training with PyTorch

PyTorch Lightning. Model Debugging. Model Deploying

---
## References
- [DataCamp PyTorch skill tracks](https://app.datacamp.com/learn/skill-tracks/deep-learning-in-python)
- [Practical Deep Learning for Coders and Deep Learning Foundations to Stable Diffusion series](https://course.fast.ai/)
- [IBM AI Engineering with Python, PyTorch & TensorFlow Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer)
- [PyTorch Ultimate 2024 - From Basics to Cutting-Edge](https://www.coursera.org/specializations/packt-pytorch-ultimate-2024---from-basics-to-cutting-edge)
- [Computer Vision Specialization by Tom Yeh Professor](https://www.coursera.org/specializations/computer-vision-cu)
---

## ⚙️ Tech Stack
- [PyTorch](https://pytorch.org/) – Deep Learning framework  
- [Torchvision](https://pytorch.org/vision/stable/index.html) – Image datasets & transforms  
- [Scikit-learn](https://scikit-learn.org/) – Preprocessing & evaluation metrics  
- [Matplotlib / Seaborn](https://matplotlib.org/) – Visualization  
- [Pandas & Numpy](https://pandas.pydata.org/) – Data manipulation  


