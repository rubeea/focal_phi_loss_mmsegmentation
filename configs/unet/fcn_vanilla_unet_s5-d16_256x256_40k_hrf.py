_base_ = [
    '../_base_/models/fcn_vanilla_unet_s5-d16.py', '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
test_cfg = dict(crop_size=(256, 256), stride=(170, 170))
evaluation = dict(metric='mIoU')
