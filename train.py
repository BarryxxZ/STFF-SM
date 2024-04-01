# -*-coding:utf-8-*-
# @author: Yiqin Qiu
# @email: barryxxz6@gmail.com

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ast
import random
import time
import argparse
from joblib import Parallel, delayed
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from sklearn.metrics import recall_score, accuracy_score
from tensorflow_addons.optimizers import AdamW
from model import STFF_SM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_file_list(file_dir, train_num, val_num, test_num):
    train_list = []
    val_list = []
    test_list = []
    for folder in file_dir:
        file_list = []
        for file in os.listdir(folder['folder']):
            file_list.append((os.path.join(folder['folder'], file), folder['class']))
        train_list += file_list[: train_num // 4]
        val_list += file_list[train_num // 4: (train_num + val_num) // 4]
        test_list += file_list[-test_num // 4:]
    return train_list, val_list, test_list


def read_files(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    array = []
    for line in lines:
        row = [float(item) for item in line.split('\t')]
        for i in range(4):
            row[i + 4] = row[i + 4] + 2
        array.append(row)
    return array


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train(arg, x_train_data, y_train_data, x_val_data, y_val_data, x_test_data, y_test_data, weight_path, model):
    y_train = to_categorical(y_train_data, num_classes=2)
    y_val = to_categorical(y_val_data, num_classes=2)
    checkpoint = ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=0, save_best_only=True,
                                 mode='max', save_weights_only=True)
    acc_print_callback = LambdaCallback(
        on_epoch_end=lambda batch, logs:
        print('acc on test set: %.4f' %
              accuracy_score(y_test_data, np.argmax(model.predict(x_test_data, batch_size=arg.batch_size), axis=1))))
    fpr_print_callback = LambdaCallback(
        on_epoch_end=lambda batch, logs:
        print('fpr on test set: %.4f' %
              (1 - recall_score(y_test_data, np.argmax(model.predict(x_test_data, batch_size=arg.batch_size), axis=1),
                                pos_label=0))))
    fnr_print_callback = LambdaCallback(
        on_epoch_end=lambda batch, logs:
        print('fnr on test set: %.4f' %
              (1 - recall_score(y_test_data, np.argmax(model.predict(x_test_data, batch_size=arg.batch_size), axis=1)))))
    callbacks_list = [checkpoint, acc_print_callback, fpr_print_callback, fnr_print_callback]

    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)
    loss = CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    print('[INFO] Training started')
    model.fit(x_train_data, y_train, batch_size=arg.batch_size, epochs=arg.epoch, validation_data=(x_val_data, y_val),
              callbacks=callbacks_list, verbose=2)
    print('[INFO] Training finished')


def test(arg, in_shape, x_test_data, y_test_data, weight_path):
    print('[INFO] Testing the detection performance')
    net = STFF_SM(in_shape)
    net.load_weights(weight_path)
    start = time.time()
    y_predict = net.predict(x_test_data, batch_size=arg.batch_size)
    end = time.time()
    print('Inference time: %.6fs' % (end - start))

    y_predict = np.argmax(y_predict, axis=1)
    print('* accuracy on test set: %0.2f%%' % (accuracy_score(y_test_data, y_predict) * 100))
    # y_predict = (y_predict > 0.5)
    tpr = recall_score(y_test_data, y_predict)
    tnr = recall_score(y_test_data, y_predict, pos_label=0)
    fpr = 1 - tnr
    fnr = 1 - tpr
    print('* FPR on test set: %0.2f' % (fpr * 100))
    print('* FNR on test set: %0.2f' % (fnr * 100))

    print('[INFO] Writing the results to the log')
    f = open("./results/result_STFF-SM_log.txt", 'a')
    f.writelines(["\n" + weight_path +
                  " Accuracy %0.2f  " % (accuracy_score(y_test_data, y_predict) * 100) +
                  "FPR %0.2f  " % (fpr * 100) + "FNR %0.2f  " % (fnr * 100)])
    f.close()
    print('[INFO] Finished')


def main(arg):
    FOLDERS = [
        {"class": 0,
         "folder": "./dataset/{}/{}/TXT/Chinese/{}s/0".format(arg.domain, arg.method, arg.length)},
        {"class": 0,
         "folder": "./dataset/{}/{}/TXT/English/{}s/0".format(arg.domain, arg.method, arg.length)},
        {"class": 1,
         "folder": "./dataset/{}/{}/TXT/Chinese/{}s/{}".format(arg.domain, arg.method, arg.length, arg.em_rate)},
        {"class": 1,
         "folder": "./dataset/{}/{}/TXT/English/{}s/{}".format(arg.domain, arg.method, arg.length, arg.em_rate)}
    ]

    print('[INFO] Dataset folders:')
    print(FOLDERS)

    model_path = './model_weights/weights_STFF-SM_{}_{}s_{}.h5'.format(arg.method, arg.length, arg.em_rate)
    print('[INFO] The path of model weight:')
    print(model_path)

    print('[INFO] Loading dataset')
    train_file, val_file, test_file = get_file_list(FOLDERS, arg.train_num, arg.val_num, arg.test_num)

    x_train = np.array(Parallel(n_jobs=6)(delayed(read_files)(item[0]) for item in train_file))
    y_train_ori = np.array([item[1] for item in train_file])

    x_val = np.array(Parallel(n_jobs=6)(delayed(read_files)(item[0]) for item in val_file))
    y_val_ori = np.array([item[1] for item in val_file])

    x_test = np.array(Parallel(n_jobs=6)(delayed(read_files)(item[0]) for item in test_file))
    y_test_ori = np.array([item[1] for item in test_file])
    print('[INFO] Loading finished')

    print('[INFO] The property of the dataset')
    print("train num: %d" % len(x_train))
    print("val num: %d" % len(x_val))
    print("test num: %d" % len(x_test))

    tmp = 0
    for item in y_train_ori:
        if item == 0:
            tmp += 1
    print("ratio in train: %.2f" % (tmp / (len(y_train_ori) - tmp)))
    tmp = 0
    for item in y_val_ori:
        if item == 0:
            tmp += 1
    print("ratio in val: %.2f" % (tmp / (len(y_val_ori) - tmp)))
    tmp = 0
    for item in y_test_ori:
        if item == 0:
            tmp += 1
    print("ratio in test: %.2f" % (tmp / (len(y_test_ori) - tmp)))

    in_shape = x_train.shape[1:]

    print('[INFO] Loading model')
    model = STFF_SM(in_shape)
    model.summary()
    if arg.train:
        train(arg, x_train, y_train_ori, x_val, y_val_ori, x_test, y_test_ori, model_path, model)
        test(arg, in_shape, x_test, y_test_ori, model_path)
    if arg.test:
        test(arg, in_shape, x_test, y_test_ori, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spatial and Temporal Fusion Model')
    parser.add_argument('--domain', type=str, default='ACB')
    parser.add_argument('--method', type=str, default='Huang')
    parser.add_argument('--length', type=str, default='1.0')
    parser.add_argument('--em_rate', type=str, default='10')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_num', type=int, default=60800)
    parser.add_argument('--val_num', type=int, default=15200)
    parser.add_argument('--test_num', type=int, default=4000)
    parser.add_argument('--train', type=ast.literal_eval, default=True,
                        help='Whether to train the model.')
    parser.add_argument('--test', type=ast.literal_eval, default=True,
                        help='Whether to test the model.')
    args = parser.parse_args()

    set_seed(args.seed)

    main(args)
