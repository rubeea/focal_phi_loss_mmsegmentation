_base_ = './fcn_r50-d8_512x512_20k_voc12aug.py'
model = dict(pretrained='open-mmlab://resnet101_v1c',
             backbone=dict(depth=101),
             decode_head=dict(num_classes=2),
             auxiliary_head=dict(num_classes=2)
             )

dataset_type = 'PLDUDataset'  # Dataset type, this will be used to define the dataset.
data_root = '../data/pldu/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='PLDUDataset',
        data_root='../data/pldu/',
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        split=None,
        ),
    val=dict(
        type='PLDUDataset',
        data_root='../data/pldu/',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        split=None,
        ),
    test=dict(
        type='PLDUDataset',
        data_root='../data/pldu/',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        split=None,
        )
)

