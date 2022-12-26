class SSD:
    _defaults = {
        'model_path': 'model_data/ssd_weights.pth',
        'classes_path': 'model_data/voc_classes.txt',
        'input_shape': [300, 300],
        'backbone': 'vgg',
        'confidence': 0.5,
        'nms_iou': 0.45,
        'anchors_size': [30, 60, 111, 162, 213, 264, 315],
        'letterbox_image': False,
        'cuda': True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self):
        pass
