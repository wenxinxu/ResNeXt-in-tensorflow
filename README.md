# ResNeXt in Tensorflow

This is an implementation of [ResNext](https://arxiv.org/abs/1611.05431) in tensorflow. The tensorboard visualization of the detailed model structure (ResNeXt-29, 4x64d as example) is shown in [graph](https://github.com/wenxinxu/ResNeXt-in-tensorflow/blob/master/figure/graph.png), [block](https://github.com/wenxinxu/ResNeXt-in-tensorflow/blob/master/figure/graph_block.png), and [split](https://github.com/wenxinxu/ResNeXt-in-tensorflow/blob/master/figure/graph_block_split.png)

I used the method in Figure 3b of the paper to implement, as there is no grouped convolution functions in the current version of tensorflow yet. (I am going check the difference of speed between Figure 3b and Figure 3c. If Figure 3b is dramatically slower, I will see if I can implement grouped convolutions myself.)

## Quick guide

You can run cifar10_train.py to train a ResNext in cifar10 and see how it works from the screen output (the code will download the data for you if you don't have it yet). It’s better to speicify version identifier before running, since the training logs, checkpoints, and error.csv file will be saved in the folder with name logs_$version. 

`python cifar10_train.py --version='test'`

You may alter the shape of the model via the hyper-parameters. Typically a ResNeXt is represented as 'ResNeXt-a, b*c'. 

a is the total layer, which is defined by 9 * FLAGS.num_resnext_blocks + 2.
b is the cardinality, which is defined by FLAGS.cardinality + 2.
c is the number of channels in each split, which is defined by FLAGS.block_unit_depth

To play with these hyper-parameters, you could either change inside the hyper-parameters.py or use the following commands:
```
python cifar10_train.py --version='test',num_resnext_blocks=3,cardinality=4
```
More explanations about the hyper-parameters can be found [here](https://github.com/wenxinxu/resnet-in-tensorflow#hyper-parameters)

## TODO:
1. Train the model and generate the learning curve
2. Figure out how much slower the 'split-concatenate' method is than the grouped convolutions. (Compare Figure 3b to Figure 3c) 
