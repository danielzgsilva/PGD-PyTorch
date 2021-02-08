from argparse import ArgumentParser
import os
from shutil import copyfile


def main(split_path):

    # load up the relevant files
    images_path = os.path.join(split_path, 'images')
    annotations_file = 'annotations.txt'
    annotations_file_path = os.path.join(split_path, annotations_file)

    with open(file=annotations_file_path, mode='r') as file:
        for i, line in enumerate(file):
            print(f'{i+1}')
            img_name = 'ILSVRC2012_val_{}.JPEG'.format(str(i+1).zfill(8))
            class_id = line.split('\t')
            class_id = class_id.pop().split('\n')[0]

            # make class id folder
            class_path = os.path.join(split_path, 'validation', class_id)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

            # move image file
            # try:
            from_path = os.path.join(images_path, img_name)
            to_path = os.path.join(class_path, img_name)
            print(from_path, to_path)
            copyfile(from_path, to_path)
            # except Exception as e:
            #     print(f'skipping... {from_path}')

def main2lol(split_path):
    # load up the relevant files
    images_path = os.path.join(split_path, 'images')
    annotations_file = 'val.txt'
    annotations_file_path = os.path.join(split_path, annotations_file)

    with open(file=annotations_file_path, mode='r') as file:
        for i, lines in enumerate(file):
            print(f'{i+1}')
            # img_name = 'ILSVRC2012_val_{}.JPEG'.format(str(i+1).zfill(8))
            line = lines.split()
            image_name = line[0]
            label = line[1]

            # make class id folder
            class_path = os.path.join(split_path, 'caffeValidation', label)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

            # move image file
            # try:
            from_path = os.path.join(images_path, image_name)
            to_path = os.path.join(class_path, image_name)
            print(from_path, to_path)
            copyfile(from_path, to_path)
            # except Exception as e:
            #     print(f'skipping... {from_path}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-split_path', help='/path/to/split (test, train, val). Folder must have an images folder and annotations.txt')

    args = parser.parse_args()

    # main(args.split_path)
    main2lol(args.split_path)