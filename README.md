# Emirati-Sign-Language-Detection
Using Deep Learning to Classify Emirati Sign Language.

This project aims to tackle the problems faced by the population not understanding sign language, by giving a direct translation of each sign made by the speaker. Artificial Intelligence will be used to classify each sign to its respective word.
The dataset to be used was created by myself. It includes 5 labelled classes (hundred, mother, ninety-two, relax, and salute), with each class having 166 video entries. 
The dataset is considered small for a real-world application. The number of classes is small, and the number of videos per class can preferably be higher. However, for the goal of this project, it should prove sufficient to see how well the model can deal with such a task and if it can differentiate between the classes. 

Neural networks require inputs in the shape of a tensor, such as a 2D image. However, for the case of a time-series data such as a video, it poses a new hurdle. The input will need to be pre-processed in a way that a neural network would be able to accept it. The pre-processing algorithm of choice to convert a video to an image that can be converted to a tensor array, while still retaining information from all frames, is the difference of frames algorithm. This project will utilize the algorithm mentioned above to convert the video instances to photos to be fed to a neural network.

![image](https://github.com/MohannadJanahi/Emirati-Sign-Language-Detection/assets/71018205/fbe818ac-9c01-40af-8a30-49b628f28cde)

The above image is an example of a preprocessed video using the difference of frames algorithm (brightened for ease of view)

When dealing with a small and limited dataset, the best course of action is to use the K-fold cross-validation algorithm. Rather than splitting the dataset into training, validation, and testing, the entire dataset can be split into K folds. For each iteration, one fold will be used for training, while the remaining folds will be used for cross-validation. This method maximizes the amount of training data and often returns better results on smaller datasets. Theoretically, the best performance will be achieved when the number of folds is equal to the amount of data. However, that would be computationally expensive. The best practice is to use a number between 5 and 10. For the purpose of this project, 7 folds are chosen. Since the difference of frames algorithm returns a lot of dead pixels, it is wise to use L1 regularization to remove unneeded features (pixels). Regularization is applied on the 1st (convolutional) layer and 9th (dense) layer. The dense layer is the most susceptible to large weights due to the huge number of trainable parameters. The model summary is shown below.

![image](https://github.com/MohannadJanahi/Emirati-Sign-Language-Detection/assets/71018205/338df8bb-d0e4-49a1-b233-b57686e79868)

An accuracy of 99.64% is achieved when using K-fold. The number of data used is now much higher, with 166 cases per class, compared to 16 cases without K-fold. Training this algorithm is generally computationally expensive, and it should ideally be used with a powerful Graphical Processing Unit (GPU).
![image](https://github.com/MohannadJanahi/Emirati-Sign-Language-Detection/assets/71018205/014747e6-a623-4f62-b7a1-e51479fb5f41)


#Future Work

In a real-world scenario with more than 5000 classes and around a million videos, the selected model in this research might not be the best choice, and transfer learning might perform better, as more classes mean more complexity. Regardless, using the difference of frames algorithm to prepare data for sign language prediction through deep learning is fruitful.

The difference of frames algorithm has one major flaw, it expects the background to be static. In the dataset, that was true. However, in a practical scenario, that will not always be the case. If the background is not static, then that motion will be captured with the algorithm as well. The same can be said about the person who is performing the hand gestures, as the algorithm expects to capture only hand movement. Any movement of the body or the head will be captured by the algorithm. This can be mitigated by using the Google MediaPipe Holistic model to track the hands, and then draw bounding boxes that cover the range of positions taken by the hand. The difference of frames algorithm can then only be applied within the bounding boxes. Any pixel outside of the bounding boxes can be blacked out. That way, the effect of a dynamic background and noise capture is reduced.
