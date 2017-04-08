# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/ResNeXt-in-tensorflow
# ==============================================================================
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

## The following flags define hyper-parameters that specifically characterize ResNeXt

tf.app.flags.DEFINE_integer('cardinality', 2, '''Cadinality, number of paths in each block''')
tf.app.flags.DEFINE_integer('block_unit_depth', 64, '''the depth of each split. 64 for cifar10
in Figure 7 of the paper''')
tf.app.flags.DEFINE_string('bottleneck_implementation', 'b', '''To use Figure 3b or 3c to
implement''')


## The following flags are related to save paths, tensorboard outputs and screen outputs

tf.app.flags.DEFINE_string('version', 'v0b3_cont', '''A version number defining the directory to
save
logs and checkpoints''')
tf.app.flags.DEFINE_integer('report_freq', 391, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')


## The following flags define hyper-parameters regards training

tf.app.flags.DEFINE_integer('train_steps', 40000, '''Total steps that you want to train''')
tf.app.flags.DEFINE_boolean('is_full_validation', True, '''Validation w/ full validation set or
a random batch''')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '''Train batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size', 125, '''Validation batch size, better to be
a divisor of 10000 for this task''')
tf.app.flags.DEFINE_integer('test_batch_size', 125, '''Test batch size''')

tf.app.flags.DEFINE_float('init_lr', 0.001, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.001, '''How much to decay the learning rate each
time''')
tf.app.flags.DEFINE_integer('decay_step0', 40000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 40000, '''At which step to decay the learning rate''')


## The following flags define hyper-parameters modifying the training network
tf.app.flags.DEFINE_integer('num_resnext_blocks', 3, '''How many blocks do you want,
total layers = 3n + 2, the paper used n=3, 29 layers, as demo''')
tf.app.flags.DEFINE_float('weight_decay', 0.0007, '''scale for l2 regularization''')


## The following flags are related to data-augmentation

tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
each side of the image''')


## If you want to load a checkpoint and continue training

tf.app.flags.DEFINE_string('ckpt_path', 'logs_v0b3/model.ckpt-79999', '''Checkpoint
directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', True, '''Whether to load a checkpoint and continue
training''')
tf.app.flags.DEFINE_string('test_ckpt_path', 'model_110.ckpt-79999', '''Checkpoint
directory to restore''')


train_dir = 'logs_' + FLAGS.version + '/'
