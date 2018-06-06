import argparse
import pickle
import os
import os.path
import numpy as np
import tensorflow as tf
import time
import random
import pickle
import pdb
import gan
import cv2
import lmdb
from  data import visualize_sample, visualize_generator, export_images
from tqdm import tqdm
tfgan = tf.contrib.gan
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""---------------------------------------------------------"""
parser = argparse.ArgumentParser()


parser.add_argument('--checkpoint', type=str, default='')
#/home/hongsi96/lsun/output_savemodel/model/model_12000.ckpt
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--noise_dims', type=int, default=100)
parser.add_argument('--output', type=str, default='test')
parser.add_argument('--data', type=str, default='data')

parser.add_argument('--lmdb_path', type=str, help='The path to the lmdb database folder. '
                                    'Support multiple database paths.')
parser.add_argument('--out_dir', type=str, default='')
parser.add_argument('--total_num', type=int, default='3033042')#3033042 #126227
parser.add_argument('--sub_num', type=int, default='120000')#120000 #3000

args = parser.parse_args()

"""---------------------------------------------------------"""

batch_size=args.batch_size
noise_dims=args.noise_dims
directory =args.output
"""---------------------------------------------------------"""


#file setting
if not os.path.exists(directory):
    os.makedirs(directory)

data_dir = args.data
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

DATA_ADD=os.getcwd()+'/'+data_dir+'/'
if os.stat(DATA_ADD).st_size == 0:
    print('need to down load file')
    pdb.set_trace()
    export_images(args.lmdb_path, args.out_dir, args.total_num, args.sub_num)


# Prepare data
print('...Data loading...')
data_list=[]
for i in tqdm(range(len(os.listdir(DATA_ADD)))):
    filename=DATA_ADD+os.listdir(DATA_ADD)[i]
    with open(filename, 'rb') as f:
        data=pickle.load(f,encoding='latin1')
        data_list.append(data)
print('...Data loaded!...')


def batch(batch_size):
    num=len(data_list)
    index = random.randrange(0,num)
    index_list=random.sample(range(len(data_list[index])),batch_size)
    batch=[]
    for i in index_list:
        batch.append(data_list[index][i])
    batch=np.array(list(batch)).astype("float64")
    batch-=128.
    batch/=128.
    return batch 


# create model
def generator_fn(noise, reuse):
  images = gan.generator(noise, reuse)
  return images

def discriminator_fn(img, reuse):
    logits = gan.discriminator(img,reuse)
    return logits



#Build Networks
# Network Inputs
tf.reset_default_graph()
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dims])
real_image_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

# Build Network
##Generator Network
gen_sample = generator_fn(noise_input, reuse=False)

##Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator_fn(real_image_input, reuse=False)
disc_fake = discriminator_fn(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real, disc_fake], axis=0)
disc_concat=tf.reshape(disc_concat,shape=[-1])

# Build the stacked generator/discriminator
stacked_gans = discriminator_fn(gen_sample, reuse=True)
stacked_gans = tf.reshape(stacked_gans,shape=[-1])
# Build Targets (real or fake images)
disc_target = tf.placeholder(tf.float32, shape=[None])
gen_target = tf.placeholder(tf.float32, shape=[None])




# Test generator
#with tf.variable_scope('Generator', reuse=True):
eval_images = generator_fn(tf.random_normal([batch_size, noise_dims]), reuse=True)
generated_data_to_visualize = tfgan.eval.image_reshaper(
    eval_images[:80,...], num_cols=10)


'''loss'''
# Build Loss

disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_target,logits=disc_concat))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gen_target, logits=stacked_gans))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)


# Generator Network Variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')


# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# save model 
saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    saver.restore(sess, args.checkpoint)
    print("Model restored.")
    start_time = time.time()
    print('')
    print('image generating...')
    for i in range(10):
        start_time = time.time()
        batch_x=batch(batch_size)
        digits_np = sess.run(generated_data_to_visualize)       
        visualize_generator(i, start_time, digits_np, directory)

    
#pdb.set_trace()




