# Haensel-Challenge


Through basic descriptive statistics of the data, I saw that there were features with all zeroes i.e standard deviation was zero. So, I dropped those columns (fields_to_drop)

I also made sure that there are no same features columns in the data.

Continuous features columns were normalized with mean of each column and standard deviation of each feature column. One of the reasons for scaling the data is to ensure that all variables are seen as equally important by smoothing out feature to feature variations. Pearson correlation have been performed and observed that the continuous are not highly correlated together. The Pearson correlation coefficient is a measure of the strength of the linear relationship between two variables.

There were 3 columns which had high deviations i.e. the outliers. In order to smooth out the outliers which can be seen as noise, I binned the values with low frequency across the rows.

Then one hot encoding was performed on those 3 columns. One-hot encoding breaks up the ranking of the feature values. That is the columns are not seen as the ordered data. If these features happened to be ordinal data then one-hot encoding should not be performed.
Plot of the class distribution can be constructed where Class “C” turns out to majority class and the rest are in the minority samples. In order to combat class imbalance problem, applying over-sampling of the minority class samples by SMOTE might be an option. SMOTE algorithm operates in 'feature space' rather than 'data space'. It creates synthetic samples by oversampling the minority class which tends to generalize better. Smote is described in this paper http://www.jair.org/media/953/live-953-2037-jair.pdf

Machine Learning Algorithm:
I’ve used simple Multi-layer perceptron with 3 hidden layers. They have non-linear decision boundaries across feature-space so having multiple stacks of hidden layers leads to more expressiveness. Neural nets can perform very well on the large scale data. Larger the scale of the data, better will be their performance. As they have lot of hyper-parameters, putting significant efforts on fine-tuning those hyper-parameters may give better performance than the other classifiers.
Neural nets have lot of hyper-parameter to tune so it is also big disadvantage. Neural nets are under active area of research and are seen as “black box”. It means that we don’t know why they do work better. Therefore, many businesses can be hesitant to apply neural nets in their business products/applications because they are “black box”.
Dropout has been applied in order to avoid any possible overfitting. Overfitting occurs when the training loss is decreasing but the validation loss seems to be increasing. In this case, the model fails to generalize well to the unseen dataset.
Since the problem is of class imbalance, accuracy is not a good metrics. I’ve applied confusion metrics, precision, recall and f1 score as metrics to see if the neural net is performing good enough. There have been several disadvantages for me in this assignment. I don’t have access to any GPUs to speed-up the neural nets. Normally I’d use AWS GPU g2nxlarge instance but for this month I’ve been having payment issues so at the moment my AWS account is disabled. My PC was not functioning when training the neural net on SMOTE. Because SMOTE resulted in oversampling of the minority class samples which means that there now many more data points in the training set. I’ve applied stratification to split the data into training and validation sets. In this way, the class label distribution is same in both the data sets. All the predicted labels belong to “C” class. But using resampled data by SMOTE during training may have helped.

I’ve written codes to optimize the tensorflow graph during the inference to bring speed up when making predictions. Weights are no longer variables but are constants during the inference. This also enables to reduce the operations by the concept of fusing. https://www.youtube.com/watch?v=JOksFH3vQgk

I’ve used the inference optimization tensorflow codes from here https://stackoverflow.com/questions/45382917/how-to-optimize-for-inference-a-simple-saved-tensorflow-1-0-1-graph

I applied (n_jobs=-1) i.e. use all the mult-threads to enable speed up of the SMOTE processing. Generator function has been used in get_batches() which brings memory efficiency. Batches are removed from the memory after each iteration.

I’ve used weights with Xavier initialization. It is quite popular, gives out good results generally and is found to be as default in the dense layers of the keras and tensorflow framework. Weights need to be initialized with the small random values. Here is the full explanation on the weight initialization http://cs231n.github.io/neural-networks-2/#init

For ReLU non-linearities, some people like to use small constant value such as 0.01 for all biases because this ensures that all ReLU units fire in the beginning and therefore obtain and propagate some gradient.

Guidelines on hyper-parameter tuning though these were not tuned at all.
If you have too large batch size then that means there will be fewer weight updates in each epoch. Therefore, you have to increase the number of epochs so that the model converges. If you use higher batch size and do not increase the number of epochs, accuracy level might seem lower and that is because the model has not converged (model still has the capacity to learn more).

This video explains about mini-batch training and gradient descent
https://www.youtube.com/watch?v=hMLUgM6kTp8&index=20&list=PLAwxTw4SYaPn_OWPFT9ulXLuQrImzHfOV

This video discusses the intuition behind tuning the value of batch size https://www.youtube.com/watch?v=GrrO1NFxaW8

This video discusses the intuition behind tuning the value of learning rate
https://www.youtube.com/watch?v=HLMjeDez7ps
