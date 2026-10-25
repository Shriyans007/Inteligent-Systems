"""A small ResNet-encoder U-Net that finds individual characters in a line of text.

Why this exists
---------------
Connected components and contours can only box ink that is physically
separate. When two letters share a stroke (the crossbar of "tt" in
"butterfly") or one letter is written over another (an "a" over an "e"), the
ink is one blob and no box-cutting rule can reliably split it. This network
looks at the *shape* of the strokes and marks where each character is, even
when the ink is shared.

Input : (H, W, 1) float32 in [0, 1], white ink on black. H and W are free
        (fully convolutional) but must be multiples of 16 -- use
        ``pad_to_multiple`` before calling the model.
Output: (H/2, W/2, 2) in [0, 1]
          channel 0  character-centre heat-map (a peak at each character)
          channel 1  half-width of the character at that peak, as a fraction
                     of LINE_HEIGHT (so the decoder knows how wide a crop to take)

The encoder is a ResNet (residual blocks, 4 stages) and the decoder is a
U-Net (upsampling plus skip connections). It is deliberately small
(about 0.4M parameters) so it trains and runs on a CPU.
"""
from pathlib import Path

UNET_MODEL = Path("artifacts/text/segmentation/unet_resnet.keras")
LINE_HEIGHT = 64     # height the network was trained on (lines are rescaled to this scale)
STRIDE = 2           # output maps are half resolution
DIVISOR = 16         # H and W must be multiples of this


def _keras():
    try:
        import keras
    except ImportError:  # very old TensorFlow
        from tensorflow import keras
    return keras


def _res_block(x, filters, stride=1):
    """Basic ResNet block: two 3x3 convs plus a (projected) shortcut."""
    L = _keras().layers
    shortcut = x
    y = L.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = L.BatchNormalization()(y)
    y = L.Activation("relu")(y)
    y = L.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = L.BatchNormalization()(y)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = L.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = L.BatchNormalization()(shortcut)
    return L.Activation("relu")(L.Add()([y, shortcut]))


def build_unet_resnet(base=16):
    """ResNet encoder (1/2, 1/4, 1/8, 1/16) plus U-Net decoder back to 1/2 resolution."""
    keras = _keras()
    L = keras.layers
    inputs = keras.Input((None, None, 1))

    x = L.Conv2D(base, 3, strides=2, padding="same", use_bias=False)(inputs)   # 1/2
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    s1 = _res_block(x, base)                       # 1/2
    s2 = _res_block(s1, base * 2, stride=2)        # 1/4
    s3 = _res_block(s2, base * 4, stride=2)        # 1/8
    s4 = _res_block(s3, base * 8, stride=2)        # 1/16
    mid = _res_block(s4, base * 8)                 # extra context for overlapping strokes

    d3 = _res_block(L.Concatenate()([L.UpSampling2D()(mid), s3]), base * 4)
    d2 = _res_block(L.Concatenate()([L.UpSampling2D()(d3), s2]), base * 2)
    d1 = _res_block(L.Concatenate()([L.UpSampling2D()(d2), s1]), base)

    # Start with a low centre probability so training is not swamped by background.
    heat = L.Conv2D(1, 1, activation="sigmoid",
                    bias_initializer=keras.initializers.Constant(-2.19))(d1)
    width = L.Conv2D(1, 1, activation="sigmoid")(d1)
    return keras.Model(inputs, L.Concatenate()([heat, width]), name="unet_resnet_chars")


def centre_loss(y_true, y_pred):
    """CenterNet-style focal loss on the heat-map plus L1 loss on the width.

    y_true has three channels: target heat-map (Gaussian peaks that reach
    exactly 1.0 at each character centre), target half-width, and a mask that
    is 1 where the width target is valid (next to each centre).
    """
    ops = _keras().ops
    heat_t, width_t, mask = y_true[..., 0], y_true[..., 1], y_true[..., 2]
    heat_p = ops.clip(y_pred[..., 0], 1e-4, 1 - 1e-4)
    width_p = y_pred[..., 1]

    pos = ops.cast(heat_t >= 0.999, "float32")
    pos_loss = -ops.log(heat_p) * (1 - heat_p) ** 2 * pos
    neg_loss = -ops.log(1 - heat_p) * heat_p ** 2 * (1 - heat_t) ** 4 * (1 - pos)
    n_pos = ops.maximum(ops.sum(pos, axis=(1, 2)), 1.0)
    focal = (ops.sum(pos_loss, axis=(1, 2)) + ops.sum(neg_loss, axis=(1, 2))) / n_pos

    n_mask = ops.maximum(ops.sum(mask, axis=(1, 2)), 1.0)
    width_loss = ops.sum(ops.abs(width_p - width_t) * mask, axis=(1, 2)) / n_mask
    return focal + 10.0 * width_loss


def load_unet(path=UNET_MODEL):
    """Load trained weights for inference (the loss is only needed for training)."""
    return _keras().models.load_model(Path(path), compile=False)
