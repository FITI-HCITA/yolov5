import os
import argparse

img_extensions = ('.jpg', 'jpeg', 'png', 'bmp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', default='data/training_cfg/test.txt', type=str, help='file name of txt')
    parser.add_argument('-p', '--pattern', type=str, help='pattern to be replace')
    parser.add_argument('-np', '--new_pattern', type=str, help='new pattern')
    args = parser.parse_args()


    file_name = args.file_name
    old_pattern = args.pattern # images/ROT_data/gc_lens-120_320x240/2023_05_28_IR
    new_pattern = args.new_pattern # ROT_4_classes/images

    with open(file_name, 'r') as f:
        lines = f.readlines()
    f = open(file_name, 'w')
    for line in lines:
        new_line = line.replace(old_pattern, new_pattern)
        f.write(new_line)

    # print(lines[0])

    # for dir_path, _, file_names in os.walk(data_folder_path):
    #     for file_name in file_names:
    #         ext_type = file_name.split('.')[-1]
    #         file_path = os.path.join(dir_path, file_name)
    #         if ext_type in img_extensions:
    #             f.write(file_path + '\n')
                
    # new_f = open('data/training_cfg/val_new.txt', 'w')
    # lines = f.readlines()
    # # new_line = lines[0].replace('./datasets', 'data/datasets')
    # # print(f'{new_line}')
    # for line in lines:
    #     new_line = line.replace('./datasets', 'data/datasets')
    #     new_f.write(new_line)
    f.close()
    # new_f.close()