import argparse
import os
import os.path as osp
import tempfile
import zipfile
from PIL import Image
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PH2 dataset to mmsegmentation format')
    parser.add_argument(
        'images_path', help='the images part of PH2 dataset')
    parser.add_argument(
        'labels_path', help='the labels part of PH2 dataset')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.out_dir is None:
        out_dir = osp.join('/content/pldu_mmsegmentation/data', 'ph2')
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

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
      print('Extracting images.zip...')
      zip_file = zipfile.ZipFile(args.images_path)
      zip_file.extractall(tmp_dir)

      now_dir = osp.join(tmp_dir, 'images')
      TRAINING_LEN = int(len(os.listdir(now_dir))*0.8) #80% of the data for training and the remainig 20% for validation
      
      print('Generating image training dataset...')
      for filename in sorted(os.listdir(now_dir))[:TRAINING_LEN]:
        img = Image.open(osp.join(now_dir, filename))
        img = img.save(osp.join(out_dir, 'img_dir', 'train',
                                    osp.splitext(filename)[0] + '.jpg'))
      
      print('Generating image validation dataset...')
      for filename in sorted(os.listdir(now_dir))[TRAINING_LEN:]:
        img = Image.open(osp.join(now_dir, filename))
        img = img.save(osp.join(out_dir, 'img_dir', 'val',
                                    osp.splitext(filename)[0] + '.jpg'))

   
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
      print('Extracting labels.zip...')
      zip_file = zipfile.ZipFile(args.labels_path)
      zip_file.extractall(tmp_dir)

      now_dir = osp.join(tmp_dir, 'gt')
      TRAINING_LEN = int(len(os.listdir(now_dir))*0.8) #80% of the data for training and the remainig 20% for validation

      print('Generating annotations for training dataset...')
      for filename in sorted(os.listdir(now_dir))[:TRAINING_LEN]:
        img = Image.open(osp.join(now_dir, filename))
                
        img = img.save(osp.join(out_dir, 'ann_dir', 'train',
                                    osp.splitext(filename)[0] + '.png'))

      print('Generating annotations for validation dataset...')
      for filename in sorted(os.listdir(now_dir))[TRAINING_LEN:]:
        img = Image.open(osp.join(now_dir, filename))
        img = img.save(osp.join(out_dir, 'ann_dir', 'val',
                                    osp.splitext(filename)[0] + '.png'))

    print('Done!')


if __name__ == '__main__':
    main()
