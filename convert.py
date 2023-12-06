""""
用于预处理数据集，使得可以被torchvision.datasets.ImageFolder直接使用
"""

import os
import shutil

def copy_from_to(file_name, source_file_path, new_folder_path):
    """
    fuc: 将file_name从source_file_path中复制一份到new_folder_path
    """

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # 拷贝后文件的路径（在新文件夹中）
    destination_file_path = os.path.join(new_folder_path, file_name)
    source_file_path = os.path.join(source_file_path, file_name)

    # 拷贝文件
    shutil.copy(source_file_path, destination_file_path)


if __name__ == '__main__':

    imgs_path = 'ILSVRC2012_img_val'
    imgs = os.listdir(imgs_path)

    labels_file = 'ILSVRC2012_validation_label.txt'
    with open(labels_file, 'r') as f:
        label_list = f.read().split('\n')

    if len(imgs) != len(label_list):
        print('NOT COMPLETE ILSVRC2012 !!!')
        print(f'{len(imgs)} images with {len(label_list)} labels in labels_file')
        exit(1)
    
    output_flodar = 'ILSVRC2012_img_val_for_ImageFolder'

    for i in range(len(imgs)):
        img, label = label_list[i].split()

        con = lambda name: '0' * (6-len(label)) + label
        output_path = os.path.join(output_flodar, con(label))
        
        print(f'copy {img} from {imgs_path} to {output_path}')
        copy_from_to(img, imgs_path, output_path)

    print('DONE !!!')


