**The full code is available [here](http://torch.ch/blog/2015/07/30/cifar.html), just clone it to your machine and it's ready to play. As a former Torch7 user, I attempt to reproduce the results from the [Torch7 post](http://torch.ch/blog/2015/07/30/cifar.html).**

My friends Wu Jun and Zhang Yujing claimed Batch Normalization[1] useless. I want to prove them wrong (打他们脸), and CIFAR-10 is a nice playground to start.

![CIFAR-10 images](http://upload-images.jianshu.io/upload_images/1231993-328c120bc6ad2d56.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

CIFAR-10 contains 60 000 labeled for 10 classes images 32x32 in size, train set has 50 000 and test set 10 000. 

The dataset is quite small by today's standards, but still a good playground for machine learning algorithms. I just use horizontal flips to augment data. One would need an NVIDIA GPU with at least 3 GB of memory.

The post and the code consist of 2 parts/files:

* model definition
* training

## The model Vgg.py

It's a VGG16-like[2] (not identical, I remove the first FC layer) network with many 3x3 filters and padding 1,1 so the sizes of feature maps after them are unchanged. They are only changed after max-pooling. Weights of convolutional layers are initialized MSR-style. Batch Normalization and Dropout are used together.

## Training train.py

That's it, you can start training:

```python
python train.py
```
The parameters with which models achieves the best performance are default in the code. I used SGD (a little out-of-date) with cross-entropy loss with learning 0.01, momentum 0.9 and weight decay 0.0005, dropping learning rate every 25 epochs. After a few hours you will have the model. The accuracy record and models at each checkpoint are saved in 'save' folder.

How accuracy improves:
![CIFAR-10 Accuracy](http://upload-images.jianshu.io/upload_images/1231993-29d2fd707a548dcd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

The best accuracy is 89.89%, removing BN or Dropout results in 88.67% and 88.73% accuracy, respectively. Batch Normalization can accelerate deep network training. Removing BN and Dropout results in 86.65% accuracy and we can observe the overfitting. 

## References

1. *Sergey Ioffe, Christian Szegedy*. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. [[arxiv]](https://arxiv.org/abs/1502.03167)
2. *K. Simonyan, A. Zisserman*. Very Deep Convolutional Networks for Large-Scale Image Recognition [[arxiv]](http://torch.ch/blog/2015/07/30/cifar.html)

