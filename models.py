import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .batch_norm import BatchNormalization
from .utils import broadcast_iou
from tensorflow.keras.utils import get_file
from params import get_model_params, IMAGENET_WEIGHTS
from initializers import conv_kernel_initializer, dense_kernel_initializer


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    # print('round_filter input={} output={}'.format(orig_f, new_filters))
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(np.ceil(multiplier * repeats))

class Swish(KL.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.nn.swish(inputs)


class DropConnect(KL.Layer):

    def __init__(self, drop_connect_rate=0., **kwargs):
        super().__init__(**kwargs)
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training=None):
        keep_prob = 1.0 - self.drop_connect_rate

        # Compute drop_connect tensor
        batch_size = tf.shape(inputs)[0]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.div(inputs, keep_prob) * binary_tensor
        return output


class SEBlock(KL.Layer):
    def __init__(self, block_args, global_params, name='seblock', **kwargs):
        super().__init__(name=name, **kwargs)
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        filters = block_args.input_filters * block_args.expand_ratio
        self.gap = KL.Lambda(lambda a: tf.reduce_mean(a, axis=[1, 2], keepdims=True))
        self.conv1 = KL.Conv2D(num_reduced_filters,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                kernel_initializer=conv_kernel_initializer,
                                padding='same',
                                use_bias=True)
        self.act1 = Swish()#KL.ReLU()
        # Excite
        self.conv2 = KL.Conv2D(filters,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                kernel_initializer=conv_kernel_initializer,
                                padding='same',
                                use_bias=True)
        self.act2 = KL.Activation('sigmoid')
    def call(self, inputs, training=False):
        x = self.gap(inputs)
        x = self.conv1(x)
        x = self.act1(x)
        # Excite
        x = self.conv2(x)
        x = self.act2(x)
        out = tf.math.multiply(x, inputs)
        return out


class MBConvBlock(KL.Layer):
    def __init__(self, block_args, global_params, drop_connect_rate=None, name='mbconvblock', **kwargs):
        super().__init__(name=name, **kwargs)
        batch_norm_momentum = global_params.batch_norm_momentum
        batch_norm_epsilon = global_params.batch_norm_epsilon
        self.has_se = (block_args.se_ratio is not None) and (
            block_args.se_ratio > 0) and (block_args.se_ratio <= 1)

        filters = block_args.input_filters * block_args.expand_ratio
        kernel_size = block_args.kernel_size
        self.block_args = block_args
        self.drop_connect_rate = drop_connect_rate
        self.conv = KL.Conv2D(filters,
                            kernel_size=[1, 1],
                            strides=[1, 1],
                            kernel_initializer=conv_kernel_initializer,
                            padding='same',
                            use_bias=False)
        self.norm = KL.BatchNormalization(axis=-1,
                                        momentum=batch_norm_momentum,
                                        epsilon=batch_norm_epsilon)
        self.act = Swish()#KL.ReLU()

        self.conv1 = KL.DepthwiseConv2D([kernel_size, kernel_size],
                                        strides=block_args.strides,
                                        depthwise_initializer=conv_kernel_initializer,
                                        padding='same',
                                        use_bias=False)
        self.norm1 = KL.BatchNormalization(axis=-1,
                                        momentum=batch_norm_momentum,
                                        epsilon=batch_norm_epsilon)
        self.act1 = Swish()#KL.ReLU()


        self.seblock = SEBlock(block_args, global_params)

        self.conv2 = KL.Conv2D(block_args.output_filters,
                            kernel_size=[1, 1],
                            strides=[1, 1],
                            kernel_initializer=conv_kernel_initializer,
                            padding='same',
                            use_bias=False)
        self.norm2 = KL.BatchNormalization(axis=-1,
                                        momentum=batch_norm_momentum,
                                        epsilon=batch_norm_epsilon)
        self.dropconnect = DropConnect(drop_connect_rate)
    def call(self, inputs, training=False):
        if self.block_args.expand_ratio != 1:
            x = self.conv(inputs)
            x = self.norm(x, training=training)
            x = self.act(x)
        else:
            x = inputs

        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.act1(x)

        if self.has_se:
            x = self.seblock(x, training=training)

        # output phase
        x = self.conv2(x)
        x = self.norm2(x, training=training)

        if self.block_args.id_skip:
            if all(s == 1 for s in self.block_args.strides) and self.block_args.input_filters == self.block_args.output_filters:
                # only apply drop_connect if skip presents.
                if self.drop_connect_rate:
                    x = self.dropconnect(x)
                x = tf.math.add(x, inputs)
        return x


class EfficientNet(tf.keras.Model):
    def __init__(self, block_args_list, global_params, include_top=True, name='efficientnet', **kwargs):
        super().__init__(name=name, **kwargs)
        batch_norm_momentum = global_params.batch_norm_momentum
        batch_norm_epsilon = global_params.batch_norm_epsilon
        self.block_args_list = block_args_list
        self.global_params = global_params
        self.include_top = include_top
        self.conv1 = KL.Conv2D(filters=round_filters(32, global_params),
                            kernel_size=[3, 3],
                            strides=[2, 2],
                            kernel_initializer=conv_kernel_initializer,
                            padding='same',
                            use_bias=False)
        self.norm1 = KL.BatchNormalization(axis=-1,
                                        momentum=batch_norm_momentum,
                                        epsilon=batch_norm_epsilon)
        self.act1 = Swish()# KL.ReLU()

        # Blocks part
        block_idx = 1
        n_blocks = sum([block_args.num_repeat for block_args in block_args_list])
        drop_rate = global_params.drop_connect_rate or 0
        drop_rate_dx = drop_rate / n_blocks

        for i,block_args in enumerate(block_args_list):
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, global_params),
                output_filters=round_filters(block_args.output_filters, global_params),
                num_repeat=round_repeats(block_args.num_repeat, global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            setattr(self, f"mbconvblock{i}", MBConvBlock(block_args, global_params,
                            drop_connect_rate=drop_rate_dx * block_idx))
            block_idx += 1

            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])

            for repeat in range(1, block_args.num_repeat):
                setattr(self, f"mbconvblock{i}_{repeat}", MBConvBlock(block_args, global_params,
                                drop_connect_rate=drop_rate_dx * block_idx))
                block_idx += 1

        # Head part
        self.conv2 = KL.Conv2D(filters=round_filters(1280, global_params),
                            kernel_size=[1, 1],
                            strides=[1, 1],
                            kernel_initializer=conv_kernel_initializer,
                            padding='same',
                            use_bias=False)
        self.norm2 = KL.BatchNormalization(axis=-1,
                                        momentum=batch_norm_momentum,
                                        epsilon=batch_norm_epsilon)
        self.act2 = Swish()# KL.ReLU()


        self.gap = KL.GlobalAveragePooling2D(data_format=global_params.data_format)
        self.dropout = KL.Dropout(global_params.dropout_rate)
        self.fc = KL.Dense(global_params.num_classes, kernel_initializer=dense_kernel_initializer)
        self.softmax = KL.Activation('softmax')
    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = self.act1(x)

        block_idx = 1
        for i,block_args in enumerate(self.block_args_list):
            x = getattr(self, f"mbconvblock{i}")(x, training=training)
            block_idx += 1
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])

            for repeat in range(1, block_args.num_repeat):
                x = getattr(self, f"mbconvblock{i}_{repeat}")(x, training=training)
                block_idx += 1

        # Head part
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.act2(x)

        if self.include_top:
            x = self.gap(x)
            if self.global_params.dropout_rate > 0:
                x = self.dropout(x)
            x = self.fc(x)
            x = self.softmax(x)
        return x


def _get_model_by_name(model_name, input_shape=None, include_top=True, weights=None, classes=1000):
    """Reference: https://arxiv.org/abs/1807.11626
    Args:
        input_shape: optional, if ``None`` default_input_shape is used
            EfficientNetB0 - (224, 224, 3)
            EfficientNetB1 - (240, 240, 3)
            EfficientNetB2 - (260, 260, 3)
            EfficientNetB3 - (300, 300, 3)
            EfficientNetB4 - (380, 380, 3)
            EfficientNetB5 - (456, 456, 3)
            EfficientNetB6 - (528, 528, 3)
            EfficientNetB7 - (600, 600, 3)
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet).
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    """
    if weights not in {None, 'imagenet'}:
        raise ValueError('Parameter `weights` should be one of [None, "imagenet"]')

    if weights == 'imagenet' and model_name not in IMAGENET_WEIGHTS:
        raise ValueError('There are not pretrained weights for {} model.'.format(model_name))

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` and `include_top`'
                         ' `classes` should be 1000')

    block_agrs_list, global_params, default_input_shape = get_model_params(
        model_name, override_params={'num_classes': classes}
    )

    model = EfficientNet(block_agrs_list, global_params, include_top=include_top)

    if weights:
        if not include_top:
            weights_name = model_name + '-notop'
        else:
            weights_name = model_name
        weights = IMAGENET_WEIGHTS[weights_name]
        weights_path = get_file(weights['name'],weights['url'],cache_subdir='models',md5_hash=weights['md5'])
        model.load_weights(weights_path)

    return model


def EfficientNetB0(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b0', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB1(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b1', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB2(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b2', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB3(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b3', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB4(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b4', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB5(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b5', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB6(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b6', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB7(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b7', include_top=include_top, weights=weights, classes=classes)

#Construst YOLO

def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = EfficientNet(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = EfficientNet(x, filters, 1)
        x = EfficientNet(x, filters * 2, 3)
        x = EfficientNet(x, filters, 1)
        x = EfficientNet(x, filters * 2, 3)
        x = EfficientNet(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = EfficientNet(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = EfficientNet(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = EfficientNet(x, filters * 2, 3)
        x = EfficientNet(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.5
    )

    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels])

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels])

    x_8, x = EfficientNet(name='yolo_efficientnet')(x)

    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(broadcast_iou(
            pred_box, true_box_flat), axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss
