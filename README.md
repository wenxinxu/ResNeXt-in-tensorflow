# ResNeXt in Tensorflow

This is an implementation of [ResNeXt](https://arxiv.org/abs/1611.05431) in tensorflow. The tensorboard visualization of the detailed model structure (ResNeXt-29, 4x64d as example) is shown in [graph](https://github.com/wenxinxu/ResNeXt-in-tensorflow/blob/master/figure/graph.png), [block](https://github.com/wenxinxu/ResNeXt-in-tensorflow/blob/master/figure/graph_block.png), and [split](https://github.com/wenxinxu/ResNeXt-in-tensorflow/blob/master/figure/graph_block_split.png). 

I implemented the blocks with both methods in Figure 3b (split->concat). 

**Figure 3c (grouped convolutions) is not available yet. ** 

tensorflow 1.0.0 is needed here. They re-named lots of functions, so the previous versions won't work...

## Quick guide

You can run cifar10_train.py to train a ResNeXt on cifar10 and see how it works from the screen output (the code will download the data for you if you don't have them yet). It’s better to speicify a version identifier before running, since the training logs, checkpoints, and error.csv file will be saved in the folder with name logs_$version. 

`python cifar10_train.py --version='test'`

You may alter the shape of the model via the hyper-parameters. Typically a ResNeXt is represented as 'ResNeXt-a, b*c'. 

a is the total layer, which is defined by 9 * FLAGS.num_resnext_blocks + 2.

b is the cardinality, which is defined by FLAGS.cardinality.

c is the number of channels in each split, which is defined by FLAGS.block_unit_depth

To play with these hyper-parameters, you could either change inside the hyper-parameters.py or use the following commands:
```
python cifar10_train.py --version='test',num_resnext_blocks=3,cardinality=4
```
More explanations about the hyper-parameters can be found [here](https://github.com/wenxinxu/resnet-in-tensorflow#hyper-parameters)

### Files included
1. hyper-parameters.py defines the hyper-parameters related to train, ResNeXt structure, data augmentation, etc.

2. cifar10_input.py includes the data I/O, pre-processing of images and data augmentation

3. resNeXt.py is the main body of ResNeXt network

4. cifar10_train.py is responsible for the training and validation

## TODO:
1. Train the model and generate the learning curve
2. Data augmentation: cv2 is not compatible w/ tensorflow 1.0.0, so I need to:

    a. Wait for a new version of opencv

    or

    b. Use the queueRunner in tensorflow as data I/O and implement data augmentaion

