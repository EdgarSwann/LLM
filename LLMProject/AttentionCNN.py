# ============================
# Blockage Detection — single notebook
# ============================
# Put this in a Jupyter cell (or split into cells).
import os
import sys
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from pathlib import Path
import matplotlib.pyplot as plt

print("TF version:", tf.__version__)

# ----------------------------
# CONFIG (adapted from train_cfg.py)
# ----------------------------
class CONFIG_TRAIN:
    APPLICATION = 'blockage_detection'
    EXPERIMENT_NAME = "japan_recs_included_no_weights"
    ROBUST = False

    NUM_CLASSES = 4

    # --- training / validation directories (change if needed) ---
    japan_train_im_path = r"C:\Users\uig41394\Desktop\Final Verison Revised\Training\images"
    japan_train_label_path = r"C:\Users\uig41394\Desktop\Final Verison Revised\Training\label_matrices"

    japan_val_im_path = r"C:\Users\uig41394\Desktop\Final Verison Revised\Validation\images"
    japan_val_label_path = r"C:\Users\uig41394\Desktop\Final Verison Revised\Validation\label_matrices"

    # Use only Japan training set (as in your config)
    TRAIN_DATASET_DICT = {japan_train_im_path: japan_train_label_path}
    VAL_DATASET_DICT = {japan_val_im_path: japan_val_label_path}

    TARGET_SIZE = (464, 336)  # (Width, Height) -> note width,height ordering in original code
    DIVISOR = 255.0

    # Callbacks / training params
    SAVE_BEST_MODEL_ONLY = True
    SAVE_WEIGHTS_ONLY = False

    FACTOR_FOR_LR_REDUCE = 0.1
    PATIENCE_FOR_LR_REDUCE = 2
    MIN_DELTA_FOR_LR_REDUCE = 0.0001
    MIN_LR = 0

    MIN_DELTA_FOR_EARLY_STOPPING = 0
    PATIENCE_FOR_EARLY_STOPPING = 5

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 100
    VAL_INTERVAL = 2

    # H5 / TFLITE backbone paths (edit if you want to load a pretrained backbone)
    # H5_MODEL_PATH = r"logs/20250313-1358_inverse_weights/models/best_model.h5"  # set to None to build from tflite or from-scratch
    H5_MODEL_PATH = None
    TFLITE_MODEL_BACKBONE_PATH = r"D:\Workspace\Projects\BlockageDetection\parking_model.tflite"

CFG = CONFIG_TRAIN()

# ----------------------------
# Loss (class from your loss.py)
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class weighted_categorical_crossentropy(tf.keras.losses.Loss):
    def __init__(self, weights):
        super().__init__(name="weighted_categorical_crossentropy")
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return tf.reduce_mean(-tf.reduce_sum(self.weights * y_true * tf.math.log(y_pred), axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"weights": self.weights.numpy().tolist()})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(weights=config["weights"])

# ----------------------------
# Metrics (from your metrics.py)
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class OverallPixelAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='overall_pixel_accuracy', **kwargs):
        super(OverallPixelAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positive = self.add_weight(name="true_positive", initializer="zeros")
        self.num_of_all_pixels = self.add_weight(name="num_of_all_pixels", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size, h, w, n_classes = y_pred.shape
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_pred_onehot = tf.one_hot(y_pred_classes, depth=self.num_classes)
        self.num_of_all_pixels.assign_add(tf.cast(batch_size * h * w, tf.float32))
        self.true_positive.assign_add(tf.reduce_sum(tf.cast(y_true * y_pred_onehot, tf.float32)))

    def result(self):
        return self.true_positive / (self.num_of_all_pixels + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positive.assign(0)
        self.num_of_all_pixels.assign(0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "num_of_true_positives": int(self.true_positive.numpy()),
            "num_of_all_pixels": int(self.num_of_all_pixels.numpy())
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(num_classes=config['num_classes'])


@tf.keras.utils.register_keras_serializable()
class AverageClassPixelAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='average_class_pixel_accuracy', **kwargs):
        super(AverageClassPixelAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positive = self.add_weight(name="true_positive", shape=(num_classes,), initializer="zeros")
        self.positive_gt_pixels = self.add_weight(name="positive_gt_pixels", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_pred_onehot = tf.one_hot(y_pred_classes, depth=self.num_classes)
        # sum across batch,h,w -> per-class counts
        self.true_positive.assign_add(tf.reduce_sum(tf.cast(y_true * y_pred_onehot, tf.float32), axis=(0,1,2)))
        self.positive_gt_pixels.assign_add(tf.reduce_sum(tf.cast(y_true, tf.float32), axis=(0,1,2)))

    def result(self):
        per_class_accuracy = self.true_positive / (self.positive_gt_pixels + tf.keras.backend.epsilon())
        return tf.reduce_mean(per_class_accuracy)

    def reset_state(self):
        self.true_positive.assign(tf.zeros(shape=(self.num_classes,)))
        self.positive_gt_pixels.assign(tf.zeros(shape=(self.num_classes,)))

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "true_positives": self.true_positive.numpy().tolist(),
            "pos_gt_pixels": self.positive_gt_pixels.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(num_classes=config['num_classes'])


# ----------------------------
# New model builder: CNN + Attention -> full-resolution output
# ----------------------------
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, AveragePooling2D
from tensorflow.keras.layers import Input, LayerNormalization, MultiHeadAttention, Reshape, TimeDistributed
from tensorflow.keras.layers import UpSampling2D, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

def get_model_with_attention(h5_model_path=None, tflite_model_path=None, cfg=CFG):
    """
    Builds a CNN encoder -> attention -> CNN decoder segmentation model.
    Returns a model that outputs full-resolution segmentation with shape:
      (batch, height, width, cfg.NUM_CLASSES)
    If h5_model_path exists, loads it (with custom objects).
    NOTE: tflite backbone loading is omitted for safety (you can add it back if needed).
    """
    # If pretrained H5 available -> load
    if h5_model_path and os.path.exists(h5_model_path):
        print("Loading model from H5:", h5_model_path)
        custom_objects = {
            'weighted_categorical_crossentropy': weighted_categorical_crossentropy,
            'AverageClassPixelAccuracy': AverageClassPixelAccuracy,
            'OverallPixelAccuracy': OverallPixelAccuracy
        }
        model = tf.keras.models.load_model(h5_model_path, custom_objects=custom_objects, compile=False)
        return model

    # Build from scratch
    width, height = cfg.TARGET_SIZE   # cfg.TARGET_SIZE is (width, height)
    input_shape = (height, width, 3)  # (h, w, c)
    no_classes = cfg.NUM_CLASSES

    inputs = Input(shape=input_shape)           # (h, w, 3)

    # -------- encoder --------
    # Block 1
    x = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = MaxPooling2D((2,2))(x)   # /2
    x = Dropout(0.2)(x)

    # Block 2
    x = Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = MaxPooling2D((2,2))(x)   # /4
    x = Dropout(0.2)(x)

    # Block 3
    x = Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = AveragePooling2D((2,2))(x)  # /8  -> feature map spatial size = (height/8, width/8)
    x = Dropout(0.2)(x)

    # record feature-map static dims (computed from cfg)
    feat_h = cfg.TARGET_SIZE[1] // 8   # height // 8
    feat_w = cfg.TARGET_SIZE[0] // 8   # width  // 8
    # channel dimension should be known (128)
    feat_c = int(x.shape[-1])  # should be 128

    # -------- attention on patches --------
    # Reshape feature map to (batch, num_patches, feat_c)
    num_patches = feat_h * feat_w
    patches = Reshape((num_patches, feat_c))(x)   # (B, P, C)

    # Multi-head self-attention (2 layers) with residual and layer norm
    att = MultiHeadAttention(num_heads=4, key_dim=64)(patches, patches)
    att = LayerNormalization()(att + patches)

    att = MultiHeadAttention(num_heads=4, key_dim=64)(att, att)
    att = LayerNormalization()(att + patches)  # residual to original patches

    # reshape back to feature map (B, feat_h, feat_w, feat_c)
    att_map = Reshape((feat_h, feat_w, feat_c))(att)

    # combine attention features with encoder features (simple concat)
    x = Concatenate(axis=-1)([x, att_map])   # (B, feat_h, feat_w, feat_c*2)

    # reduce channels
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)

    # -------- decoder: upsample back to input size --------
    # Upsample 1: feat_h*2, feat_w*2
    x = UpSampling2D((2,2))(x)   # /4
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)

    # Upsample 2: /2 -> /2
    x = UpSampling2D((2,2))(x)   # /2
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)

    # Upsample 3: back to original size
    x = UpSampling2D((2,2))(x)   # original
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)

    # Final segmentation head: full-resolution output
    outputs = Conv2D(no_classes, (1,1), activation='softmax', name='segmentation_output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    print("Built attention-CNN segmentation model with output shape:", model.output_shape)

    return model


# ----------------------------
# Utilities: build dataset from TRAIN_DATASET_DICT / VAL_DATASET_DICT
# ----------------------------
def gather_pairs_from_dict(dataset_dict, allowed_img_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    """
    Given a dict {img_dir: label_dir, ...} returns list of (img_path, label_path)
    Matching is done via basename (filename without extension).
    """
    pairs = []
    for img_dir, label_dir in dataset_dict.items():
        img_dir = Path(img_dir)
        label_dir = Path(label_dir)
        if not img_dir.exists() or not label_dir.exists():
            print("Warning: path does not exist:", img_dir, label_dir)
            continue

        # create mapping basename -> file for labels
        label_map = {}
        for p in label_dir.iterdir():
            if p.is_file():
                label_map[p.stem] = str(p)

        # iterate images and find matching label
        for p in img_dir.iterdir():
            if p.is_file() and p.suffix.lower() in allowed_img_exts:
                stem = p.stem
                if stem in label_map:
                    pairs.append((str(p), label_map[stem]))
                else:
                    # try to find label with same prefix ignoring suffix differences
                    # (skip if no match)
                    pass
    return pairs

def make_tf_dataset_from_pairs_json(pairs, cfg=CFG, shuffle=True):
    """
    Build a tf.data.Dataset from image paths and JSON mask paths.

    - Images are resized to cfg.TARGET_SIZE (height, width)
    - Masks are loaded from JSON, resized to cfg.TARGET_SIZE
    - Returns a batched dataset yielding (image, mask)
    """
    if len(pairs) == 0:
        raise ValueError("No file pairs found for dataset!")

    img_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_paths), reshuffle_each_iteration=True)

    def _parse(img_path, mask_path):
        # --- load image ---
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32) / 255.0
        height, width = cfg.TARGET_SIZE[1], cfg.TARGET_SIZE[0]
        img = tf.image.resize(img, (height, width))

        # --- load mask from JSON ---
        mask_str = tf.io.read_file(mask_path)
        mask = tf.py_function(
            func=lambda x: tf.convert_to_tensor(json.loads(x.numpy()), dtype=tf.float32),
            inp=[mask_str],
            Tout=tf.float32
        )

        # The JSON mask is (21,29,4); resize to full resolution
        mask.set_shape([None, None, cfg.NUM_CLASSES])
        mask = tf.image.resize(mask, (height, width), method="nearest")

        return img, mask

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ----------------------------
# Prepare train/val datasets
# ----------------------------
train_pairs = gather_pairs_from_dict(CFG.TRAIN_DATASET_DICT)
val_pairs = gather_pairs_from_dict(CFG.VAL_DATASET_DICT)

print("Sample train pairs:")
for i in range(min(5, len(train_pairs))):
    print(train_pairs[i])

print("Sample val pairs:")
for i in range(min(5, len(val_pairs))):
    print(val_pairs[i])

print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

train_dataset = make_tf_dataset_from_pairs_json(train_pairs, CFG, shuffle=True)
val_dataset = make_tf_dataset_from_pairs_json(val_pairs, CFG, shuffle=False)


# ----------------------------
# Build model (load H5 if present, else build from tflite/backbone)
# ----------------------------
# If you want to start from scratch set H5_MODEL_PATH to None and ensure TFLITE path is set if needed.
h5_path = CFG.H5_MODEL_PATH if (CFG.H5_MODEL_PATH and os.path.exists(CFG.H5_MODEL_PATH)) else None
tflite_path = CFG.TFLITE_MODEL_BACKBONE_PATH if (CFG.TFLITE_MODEL_BACKBONE_PATH and os.path.exists(CFG.TFLITE_MODEL_BACKBONE_PATH)) else None
model = get_model_with_attention(h5_model_path=h5_path, tflite_model_path=tflite_path, cfg=CFG)

# ----------------------------
# Compile model with your loss & metric
# ----------------------------
loss_obj = weighted_categorical_crossentropy(weights=[1.] * CFG.NUM_CLASSES)
metric_obj = AverageClassPixelAccuracy(num_classes=CFG.NUM_CLASSES)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CFG.LEARNING_RATE),
              loss=loss_obj,
              metrics=[metric_obj])

# ----------------------------
# Callbacks and logging
# ----------------------------
# create output folder with timestamp
root_path = os.getcwd()
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
out_dir = os.path.join(root_path, "logs", now + "_" + CFG.EXPERIMENT_NAME)
os.makedirs(out_dir, exist_ok=True)

checkpoint_path = os.path.join(out_dir, "best_model.h5")
csv_log_path = os.path.join(out_dir, "training.csv")

checkpoint_cb = ModelCheckpoint(filepath=checkpoint_path,
                                monitor='val_loss',
                                save_best_only=CFG.SAVE_BEST_MODEL_ONLY,
                                save_weights_only=CFG.SAVE_WEIGHTS_ONLY)

reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss',
                                 factor=CFG.FACTOR_FOR_LR_REDUCE,
                                 patience=CFG.PATIENCE_FOR_LR_REDUCE,
                                 min_delta=CFG.MIN_DELTA_FOR_LR_REDUCE,
                                 min_lr=CFG.MIN_LR,
                                 verbose=1)

earlystop_cb = EarlyStopping(monitor='val_loss',
                             patience=CFG.PATIENCE_FOR_EARLY_STOPPING,
                             restore_best_weights=True)

csv_cb = CSVLogger(csv_log_path)
tb_cb = TensorBoard(log_dir=os.path.join(out_dir, "tensorboard"))

cbs = [checkpoint_cb, reduce_lr_cb, earlystop_cb, csv_cb, tb_cb]

# ----------------------------
# Train
# ----------------------------
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=CFG.NUM_EPOCHS,
                    callbacks=cbs)

# save final model (best is already saved by checkpoint_cb)
final_model_path = os.path.join(out_dir, "final_model.h5")
model.save(final_model_path)
print("Saved final model to", final_model_path)

# ----------------------------
# Evaluate on validation (no separate test set requested)
# ----------------------------
val_loss, val_metric = model.evaluate(val_dataset)
print("Validation loss, metric:", val_loss, val_metric)

# ----------------------------
# Plot training curves and sample predictions
# ----------------------------
plt.figure(figsize=(8,4))
plt.plot(history.history.get("loss", []), label="train_loss")
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.legend()
plt.title("Loss")
plt.show()

if "average_class_pixel_accuracy" in history.history:
    plt.figure(figsize=(8,4))
    plt.plot(history.history["average_class_pixel_accuracy"], label="train_avg_acc")
    if "val_average_class_pixel_accuracy" in history.history:
        plt.plot(history.history["val_average_class_pixel_accuracy"], label="val_avg_acc")
    plt.legend()
    plt.title("Average Class Pixel Accuracy")
    plt.show()

import matplotlib.pyplot as plt
import tensorflow as tf

# Take 1 batch from validation
for images, masks in val_dataset.take(1):
    preds = model.predict(images)  # (batch, 336, 464, 4)

    batch_size = images.shape[0]
    n_show = min(3, batch_size)  # show up to 3 examples

    plt.figure(figsize=(12, 4 * n_show))
    for i in range(n_show):
        # Input image
        plt.subplot(n_show, 3, 3*i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title("Input Image")

        # Ground truth: convert one-hot to class index
        gt = tf.argmax(masks[i], axis=-1).numpy()  # (336,464)
        plt.subplot(n_show, 3, 3*i + 2)
        plt.imshow(gt, interpolation='nearest')
        plt.axis('off')
        plt.title("Ground Truth")

        # Predicted mask: argmax over classes
        pred = tf.argmax(preds[i], axis=-1).numpy()  # (336,464)
        plt.subplot(n_show, 3, 3*i + 3)
        plt.imshow(pred, interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction")

    plt.tight_layout()
    plt.show()
    break


print("Notebook run finished. Logs and models saved to:", out_dir)
