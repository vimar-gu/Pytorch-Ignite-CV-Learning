from easydict import EasyDict as edict


_C = edict()
rpn_cfg = _C

_C.TRAIN = edict()

_C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
_C.TRAIN.RPN_POST_NMS_TOP_N = 2000
_C.TRAIN.RPN_NMS_THRESH = 0.7
_C.TRAIN.RPN_MIN_SIZE = 8

_C.TEST = edict()

_C.TEST.RPN_PRE_NMS_TOP_N = 6000
_C.TEST.RPN_POST_NMS_TOP_N = 300
_C.TEST.RPN_NMS_THRESH = 0.7
_C.TEST.RPN_MIN_SIZE = 16


_C.ANCHOR_SCALES = [8,16,32]

_C.ANCHOR_RATIOS = [0.5,1,2]

_C.FEAT_STRIDE = [16, ]

_C.DEMENSION_INPUT = 512
