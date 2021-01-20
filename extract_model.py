#! -*- coding: utf-8 -*-
# 法研杯2020 司法摘要
# 抽取式：主要模型
# 科学空间：https://kexue.fm

import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import LayerNormalization
from bert4keras.optimizers import Adam
from bert4keras.snippets import open
from keras.layers import *
from keras.models import Model
from snippets import *

# 配置信息
input_size = 768
hidden_size = 384
epochs = 20
batch_size = 64
threshold = 0.2
data_extract_json = data_json[:-5] + '_extract.json'
data_extract_npy = data_json[:-5] + '_extract.npy'

if len(sys.argv) == 1:
    fold = 0
else:
    fold = int(sys.argv[1])


def load_data(filename):
    """加载数据
    返回：[(texts, labels, summary)]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            D.append(json.loads(l))
    return D


class ResidualGatedConv1D(Layer):
    """门控卷积
    """
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(ResidualGatedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True

    def build(self, input_shape):
        super(ResidualGatedConv1D, self).build(input_shape)
        self.conv1d = Conv1D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='same',
        )
        self.layernorm = LayerNormalization()

        if self.filters != input_shape[-1]:
            self.dense = Dense(self.filters, use_bias=False)

        self.alpha = self.add_weight(
            name='alpha', shape=[1], initializer='zeros'
        )

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs = inputs * mask[:, :, None]

        outputs = self.conv1d(inputs)
        gate = K.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs

    def compute_output_shape(self, input_shape):
        shape = self.conv1d.compute_output_shape(input_shape)
        return (shape[0], shape[1], shape[2] // 2)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        }
        base_config = super(ResidualGatedConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


x_in = Input(shape=(None, input_size))
x = x_in

x = Masking()(x)
x = Dropout(0.1)(x)
x = Dense(hidden_size, use_bias=False)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=2)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=4)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=8)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
x = Dropout(0.1)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(x_in, x)
model.compile(
    loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy']
)
model.summary()


def evaluate(data, data_x, threshold=0.2):
    """验证集评估
    """
    y_pred = model.predict(data_x)[:, :, 0]
    total_metrics = {k: 0.0 for k in metric_keys}
    for d, yp in tqdm(zip(data, y_pred), desc=u'评估中'):
        yp = yp[:len(d[0])]
        yp = np.where(yp > threshold)[0]
        pred_summary = ''.join([d[0][i] for i in yp])
        metrics = compute_metrics(pred_summary, d[2], 'char')
        for k, v in metrics.items():
            total_metrics[k] += v
    return {k: v / len(data) for k, v in total_metrics.items()}


class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self):
        self.best_metric = 0.0

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(valid_data, valid_x, threshold + 0.1)
        if metrics['main'] >= self.best_metric:  # 保存最优
            self.best_metric = metrics['main']
            model.save_weights('weights/extract_model.%s.weights' % fold)
        metrics['best'] = self.best_metric
        print(metrics)


if __name__ == '__main__':

    # 加载数据
    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy)
    data_y = np.zeros_like(data_x[..., :1])

    for i, d in enumerate(data):
        for j in d[1]:
            data_y[i, j] = 1

    train_data = data_split(data, fold, num_folds, 'train')
    valid_data = data_split(data, fold, num_folds, 'valid')
    train_x = data_split(data_x, fold, num_folds, 'train')
    valid_x = data_split(data_x, fold, num_folds, 'valid')
    train_y = data_split(data_y, fold, num_folds, 'train')
    valid_y = data_split(data_y, fold, num_folds, 'valid')

    # 启动训练
    evaluator = Evaluator()

    model.fit(
        train_x,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[evaluator]
    )

else:

    model.load_weights('weights/extract_model.%s.weights' % fold)
