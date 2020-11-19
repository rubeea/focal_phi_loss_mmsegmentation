import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PLDUDataset(CustomDataset):
    """PLDU dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '_.png'.
    """

    CLASSES = ('background', 'powerline')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(PLDUDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            ignore_index=None,
            **kwargs)
        assert osp.exists(self.img_dir)
