from scipy.io import savemat
import joblib
import matplotlib.pyplot as plt
import random
import six
from six.moves import xrange
from datetime import datetime
from tensorflow import keras
import tensorflow as tf
import numpy as np
from joblib import load
import os
import tensorflow.keras.backend as K
import scipy.stats
from sklearn.preprocessing import normalize, MinMaxScaler
import scipy.io
import tensorflow.keras.backend as K
# Set random seed
from sklearn import svm,neural_network
from scipy.stats import spearmanr
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorboard.plugins.hparams import api as hp
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(max_iter=1000)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#  except RuntimeError as e:
#    # Memory growth must be set before GPUs have been initialized
#    print(e)
random_seed = 21
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
scaler = MinMaxScaler(feature_range=(-1,1))
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
features_file = load('/mnt/b9f5646b-2c64-4699-8766-c4bba45fb442/konvid/konvid_features/konvid_cornia_features.z')

full_features = features_file["features"]
features = []

for f in full_features:
    features.append(f[0:-1:20,:])
scores = np.asarray(features_file["score"],dtype=np.float32)
scores  = scores/np.amax(scores)*100
print(len(features))
print(len(scores))
print(features[0])
shuffle_in_unison(features,scores)

DATASET_SIZE = scores.shape[0]
trainval_size = int(0.9 * DATASET_SIZE)
test_size = int(0.1 * DATASET_SIZE)
val_size = int(0.1 * DATASET_SIZE)
#scores = np.expand_dims(scores,axis=1)
val_features = []
val_scores = []

train_features_o = []
train_scores_o = []
test_features_o = []
test_scores_o = []
for ind,f in enumerate(features):
    if(ind<val_size):
        val_features.append(f)
        val_scores.append(scores[ind])
    elif(ind<trainval_size):
        train_features_o.append(f)
        train_scores_o.append(scores[ind])
    else:
        test_features_o.append(f)
        test_scores_o.append(scores[ind])
    
MINLEN = len(min(features,key=len))
print(MINLEN)
VID_LENGTH = int(MINLEN/2)
def cut_vids(features,scores,length):
    cut_f = [] 
    cut_s = []
    for i,f in enumerate(features):
        cut_f.append(f[0:VID_LENGTH,:])
        cut_s.append(scores[i])
        cut_f.append(f[-VID_LENGTH:,:])
        cut_s.append(scores[i])
        rns = np.random.random_integers(0,VID_LENGTH,(2,))

        cut_f.append(f[rns[0]:rns[0]+VID_LENGTH,:])
        cut_s.append(scores[i])

        cut_f.append(f[rns[1]:rns[1]+VID_LENGTH,:])
        cut_s.append(scores[i])
    return cut_f,cut_s

def avg_features(full_features):
    features = []
    for n1,vid in enumerate(full_features):
        vid = np.asarray(vid)
    #    vid = vid[:,-1]
    #    print(vid)
        features.append(np.average(vid,axis=0))
    return features
train_features,train_scores = cut_vids(train_features_o,train_scores_o,VID_LENGTH)
val_features,val_scores = cut_vids(val_features,val_scores,VID_LENGTH)
test_features,test_scores = cut_vids(test_features_o,test_scores_o,VID_LENGTH)
train_features = np.asarray(train_features)
val_features = np.asarray(val_features)
test_features = np.asarray(test_features)
#train_features = avg_features(train_features)
#val_features = avg_features(val_features)
#test_features = avg_features(test_features)
#print(train_features.shape)
#print(val_features.shape)
print(train_features.shape)
feature_mean = np.average(train_features,axis=(0,1))
print(feature_mean.shape)
train_features = train_features-feature_mean
val_features = val_features-feature_mean
test_features = test_features-feature_mean
print(train_features.shape)
shuffle_in_unison(train_features,train_scores)
shuffle_in_unison(val_features,val_scores)


#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#for device in gpu_devices:
#    tf.config.experimental.set_memory_growth(device, True)
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_scores))
val_dataset =  tf.data.Dataset.from_tensor_slices((val_features, val_scores))
test_dataset =  tf.data.Dataset.from_tensor_slices((test_features, test_scores))
BATCH_SIZE = 200
SHUFFLE_BUFFER_SIZE = DATASET_SIZE+1
n_examples = scores.shape[0]
STEPS_PER_EPOCH = n_examples/BATCH_SIZE
#print(val_size)
# Attention GRU network       
class AttLayer(tf.keras.layers.Layer):
    def __init__(self,feature_length,**kwargs):        
        #self.input_spec = [InputSpec(ndim=3)]
        self.feature_length = feature_length
        super(AttLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W_key = self.add_weight(name='W_key', 
                                      shape=(input_shape[-1],self.feature_length),
                                      initializer='normal',
                                      trainable=True)
        
        self.W_query = self.add_weight(name='W_query', 
                                      shape=(input_shape[-1],self.feature_length,),
                                      initializer='normal',
                                      trainable=True)
        self.W_value = self.add_weight(name='W_value', 
                                      shape=(input_shape[-1],self.feature_length),
                                      initializer='normal',
                                      trainable=True)
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!
    def call(self, x, mask=None):
        keys = K.tanh(tf.matmul(x,self.W_key))
        query = K.tanh(tf.matmul(x,self.W_query))
        value = K.tanh(tf.matmul(x,self.W_value))
        
        attn_scores_full = tf.matmul(query,keys,transpose_b=True)
        attn_scores = tf.linalg.diag_part(attn_scores_full)
        attn_scores_softmax = tf.expand_dims(tf.nn.softmax(attn_scores),axis=2)
        weighted_input = tf.matmul(value,attn_scores_softmax,transpose_a=True)

        return weighted_input,attn_scores_softmax

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],), shape_b[:-1]]    # each numpy file contains 36 brisque features of n frames
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'feature_length': self.feature_length,
        })
        return config
# read each file as an array
# read the score in from another file
logdir = "logs/hparams/higrade_best"# + datetime.now().strftime("%Y%m%d-%H%M%S")

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32,64]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1,0.01,0.001,0.0001]))
HP_LEARNING_RATE = hp.HParam('lr',hp.Discrete([0.1,0.01,0.001,0.0001]))
HP_FEATURE_LENGTH= hp.HParam('feature_length',hp.Discrete([8,16,32]))
METRIC_LOSS = 'epoch_loss'

HPARAMS=[HP_NUM_UNITS,HP_DROPOUT,HP_LEARNING_RATE,HP_FEATURE_LENGTH]

def model_fn(hparams,seed):
    #    rng = random.Random(seed)
    x_train = tf.keras.Input(shape=(VID_LENGTH,20000,))
#    l_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True,dropout=0.05))(x_train)
    # l_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True,dropout=0.1))(l_gru)
    # l_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True,dropout=0.1))(l_gru)

    # # print(l_gru)

    l_att1,weights1 = AttLayer(hparams[HP_FEATURE_LENGTH])(x_train)
    l_att2,weights2 = AttLayer(hparams[HP_FEATURE_LENGTH])(x_train)
    l_att3,weights3 = AttLayer(hparams[HP_FEATURE_LENGTH])(x_train)
    l_att4,weights4 = AttLayer(hparams[HP_FEATURE_LENGTH])(x_train)
    l_att5,weights5 = AttLayer(hparams[HP_FEATURE_LENGTH])(x_train)
    l_att6,weights6 = AttLayer(hparams[HP_FEATURE_LENGTH])(x_train)
    l_att7,weights7 = AttLayer(hparams[HP_FEATURE_LENGTH])(x_train)
    l_att8,weights8 = AttLayer(hparams[HP_FEATURE_LENGTH])(x_train)
    l_att9,weights9 = AttLayer(hparams[HP_FEATURE_LENGTH])(x_train)
    l_att = tf.keras.layers.concatenate([l_att1,l_att2,l_att3,l_att4,l_att5,l_att6,l_att7,l_att8,l_att9],axis=1)
    l_att = tf.squeeze(l_att,axis=2)

#    concat(64,)
 #= tf.keras.layers.concatenate(inputs=[l_att])

    dense0 = tf.keras.layers.Dense(hparams[HP_NUM_UNITS]*4,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.005))(l_att)
    dr = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(dense0)
    dense1 = tf.keras.layers.Dense(hparams[HP_NUM_UNITS]*2,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.005))(dr)
    dr = tf.keras.layers.Dropout(hparams[HP_DROPOUT]*2)(dense1)
    dense2 = tf.keras.layers.Dense(hparams[HP_NUM_UNITS],activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.005))(dr)
#    bn = tf.keras.layers.BatchNormalization()(dense)
    dr = tf.keras.layers.Dropout(hparams[HP_DROPOUT]*4)(dense2)
    
#    drop = tf.keras.layers.Dropout(rate=0.1)(lr)
#  
    
    preds = tf.keras.layers.Dense(1, activation='linear',kernel_regularizer=tf.keras.regularizers.l2(0.005))(dr)
    print(preds.shape)
    model = tf.keras.Model(x_train, preds)
#    for layer in model.layers:
#        W = layer.get_weights()
##        print(W)
#        for w in W:
#            print(w.shape)

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      hparams[HP_LEARNING_RATE],
      decay_steps=STEPS_PER_EPOCH*500,
      decay_rate=1,
      staircase=False)

    ad_opt = tf.keras.optimizers.Adam(lr_schedule)
    model.compile(loss='mse',
                  optimizer=ad_opt)
    return model

def run(train_dataset,val_dataset, base_logdir, session_id, hparams):
    """Run a training/validation session.
    Flags must have been parsed for this function to behave.
    Args:
      data: The data as loaded by `prepare_data()`.
      base_logdir: The top-level logdir to which to write summary data.
      session_id: A unique string ID for this session.
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
    """
    model = model_fn(hparams=hparams, seed=session_id)
    logdir = os.path.join(base_logdir, session_id)

    EPOCHS =300 

    
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,profile_batch=0)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40,restore_best_weights=True)

    hparams_callback = hp.KerasCallback(logdir, hparams)
    model.fit(train_dataset, epochs=EPOCHS,callbacks=[tensorboard_callback,hparams_callback,callback],validation_data = val_dataset)

def run_all(train_dataset,val_dataset,logdir, verbose=False):
    """Perform random search over the hyperparameter space.
    Arguments:
      logdir: The top-level directory into which to write data. This
        directory should be empty or nonexistent.
      verbose: If true, print out each run's name as it begins.
    """
    rng = random.Random(0)

    with tf.summary.create_file_writer(logdir).as_default():
      hp.hparams_config(
        HPARAMS,
        metrics=[hp.Metric("batch_loss", group="train", display_name="loss (train)",),hp.Metric(METRIC_LOSS,group='validation', display_name='Val loss')],
      )

    sessions_per_group = 1
    num_sessions = 100 * sessions_per_group
    session_index = 0  # across all session groups
    for group_index in xrange(20):
        hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
        hparams_string = str(hparams)
        for repeat_index in xrange(sessions_per_group):
            session_id = str(session_index)
            session_index += 1
            if verbose:
                print(
                    "--- Running training session %d/%d"
                    % (session_index, num_sessions)
                )
                print(hparams_string)
                print("--- repeat #: %d" % (repeat_index + 1))
            run(
                train_dataset,val_dataset,
                base_logdir=logdir,
                session_id=session_id,
                hparams=hparams,
            )


train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)

test_dataset = test_dataset.batch(BATCH_SIZE)
train_scores = np.asarray(train_scores)
sroccs = []
train=False
if(train==True):
    #    drop = 0.1
    # print(model.metrics_names)


    
    run_all(train_dataset,val_dataset,logdir=logdir, verbose=True)
else:
    hparams = {HP_NUM_UNITS:32,HP_LEARNING_RATE:0.01,HP_FEATURE_LENGTH:16,HP_DROPOUT:0.01}
    model = model_fn(hparams,0)
    EPOCHS =300 
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,profile_batch=0)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40,restore_best_weights=True)
#    model = tf.keras.models.load_model('/home/labuser-admin/code/attention/saved_models/attention_viz.h5',custom_objects={'AttLayer':AttLayer}) 
    intermed_model = tf.keras.Model(model.input,outputs=[model.get_layer('att_layer').output[1],model.get_layer('att_layer_1').output[1],model.get_layer('att_layer_2').output[1],model.get_layer('att_layer_3').output[1]])
    model.fit(train_dataset, epochs=EPOCHS,callbacks=[tensorboard_callback,callback],validation_data = val_dataset)
    model.save('./saved_models/attention_viz.h5')
    sroccs=[]
    for layer in model.layers:
        print(layer.name)


    pred_list = []
    ys = []
    cut_f = []
    for i,f in enumerate(test_features_o):
        cut_f.append(f[0:VID_LENGTH,:])
        cut_f.append(f[-VID_LENGTH:,:])
        rns = np.random.random_integers(0,VID_LENGTH,(2,))

        cut_f.append(f[rns[0]:rns[0]+VID_LENGTH,:])

        cut_f.append(f[rns[1]:rns[1]+VID_LENGTH,:])
    pred_scores =[]
    test_scores_o=np.asarray(test_scores_o)
    print(test_scores_o.shape)
    for cf in cut_f:
#        test_f = np.average(cf,axis=0)
        test_f = cf - feature_mean
    
        test_f = np.expand_dims(test_f,axis=0)
        pred_score = model.predict(test_f)
        weights1,weights2,weights3,weights4 = intermed_model.predict(test_f)

#        print(weights1.shape)
#        a1 = weights1[0,:]
#        a2 = weights2[0,:]
#        a3 = weights3[0,:]
#        a4 = weights4[0,:]
#        b = cf[:,-1]
#        fig, axs = plt.subplots(5)
#        axs[0].plot(a1)
#        axs[1].plot(a2)
#        axs[2].plot(a3)
#        axs[3].plot(a4)
#        axs[4].plot(b)
#        plt.show()
#        break
        pred_scores.append(pred_score)
    pred_scores=np.asarray(pred_scores)
    pred_scores = np.reshape(pred_scores,(-1,4))
    pred_scores = np.average(pred_scores,axis=1)
#    t = {'data':pred_scores}
#    savemat('higrade_att.mat',t)
#    t = {'data':test_scores_o}
#    savemat('test_scores.mat',t)
#    joblib.dump(test_scores_o,'test_scores_3.z')
#    joblib.dump(pred_scores,'higrade_att.z')    
    srocc= scipy.stats.spearmanr(pred_scores, test_scores_o)
    print('This is for spatial w att and R')#  with attention and without RNN')
    print(srocc, ' median test SROCC') 
#model = tf.keras.models.load_model('/home/labuser-admin/code/attention/saved_models/attention_brisque_0.h5',custom_objects={'AttLayer':AttLayer})
#pred_list = []
#sroccs  = []
#ys = []
#cut_f = []
#for i,f in enumerate(train_features_o):
#    cut_f.append(f[0:VID_LENGTH,:])
#    cut_f.append(f[-VID_LENGTH:,:])
#    rns = np.random.random_integers(0,VID_LENGTH,(2,))
#
#    cut_f.append(f[rns[0]:rns[0]+VID_LENGTH,:])
#
#    cut_f.append(f[rns[1]:rns[1]+VID_LENGTH,:])
#pred_scores =[]
#train_scores_o=np.asarray(train_scores_o)
#print(train_scores_o.shape)
#for cf in cut_f:
##    test_f = np.average(cf,axis=0)
#    test_f = cf - feature_mean
#    test_f = np.expand_dims(test_f,axis=0)
#    pred_score = model.predict(test_f)
#    pred_scores.append(pred_score)
#pred_scores=np.asarray(pred_scores)
#pred_scores = np.reshape(pred_scores,(-1,4))
#pred_scores = np.average(pred_scores,axis=1)
##print(pred_scores-train_scores_o)
##print(train_scores_o)
#srocc= scipy.stats.spearmanr(pred_scores, train_scores_o)
#print('This is for spatial w att and R')#  with attention and without NN')
#
