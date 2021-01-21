import argparse
import os
import os.path as osp
import tempfile
import zipfile

import mmcv

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert HRF dataset to mmsegmentation format')
    parser.add_argument('healthy_path', help='the path of healthy.zip')
    parser.add_argument(
        'healthy_manualsegm_path', help='the path of healthy_manualsegm.zip')
    parser.add_argument('glaucoma_path', help='the path of glaucoma.zip')
    parser.add_argument(
        'glaucoma_manualsegm_path', help='the path of glaucoma_manualsegm.zip')
    parser.add_argument(
        'diabetic_retinopathy_path',
        help='the path of diabetic_retinopathy.zip')
    parser.add_argument(
        'diabetic_retinopathy_manualsegm_path',
        help='the path of diabetic_retinopathy_manualsegm.zip')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    images_path = [
        args.healthy_path, args.glaucoma_path, args.diabetic_retinopathy_path
    ]
    annotations_path = [
        args.healthy_manualsegm_path, args.glaucoma_manualsegm_path,
        args.diabetic_retinopathy_manualsegm_path
    ]
    if args.out_dir is None:
        out_dir = osp.join('data', 'ph2')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir','train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir','val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    print('Generating images...')
    for now_path in images_path:
        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
            zip_file = zipfile.ZipFile(now_path)
            zip_file.extractall(tmp_dir)

            TRAINING_LEN = len(os.listdir(tmp_dir))*0.8 #80% of the data for training and the remainig 20% for validation
            
            for filename in sorted(os.listdir(tmp_dir))[:TRAINING_LEN]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                mmcv.imwrite(
                    img,
                    osp.join(out_dir, 'img_dir','train',
                             osp.splitext(filename)[0] + '.jpg'))
            for filename in sorted(os.listdir(tmp_dir))[TRAINING_LEN:]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                mmcv.imwrite(
                    img,
                    osp.join(out_dir, 'img_dir',val',
                             osp.splitext(filename)[0] + '.jpg'))

    print('Generating annotations...')
    for now_path in annotations_path:
        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
            zip_file = zipfile.ZipFile(now_path)
            zip_file.extractall(tmp_dir)

            TRAINING_LEN = len(os.listdir(tmp_dir))*0.8 #80% of the data for training and the remainig 20% for validation

            for filename in sorted(os.listdir(tmp_dir))[:TRAINING_LEN]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                
                mmcv.imwrite(
                    img,
                    osp.join(out_dir, 'ann_dir', 'train',
                             osp.splitext(filename)[0] + '.png'))
            for filename in sorted(os.listdir(tmp_dir))[TRAINING_LEN:]:
                img = mmcv.imread(osp.join(tmp_dir, filename))
                mmcv.imwrite(
                    img,
                    osp.join(out_dir, 'ann_dir', 'val',
                             osp.splitext(filename)[0] + '.png'))

    print('Done!')


if __name__ == '__main__':
    main()
