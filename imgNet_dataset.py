import os
import random

dataset_path = '/media/root/9E46A26D46A2463D/OpenMMLab/hw1/flower_dataset'
train_path = os.path.join(dataset_path, 'train.txt')
val_path = os.path.join(dataset_path, 'val.txt')
classes_path = os.path.join(dataset_path, 'classes.txt')

# 读取类别列表
with open(classes_path, 'r') as f:
    classes = f.read().strip().split('\n')

# 创建字典映射类别名称到编号
classes_dict = {classes[i]: i for i in range(len(classes))}

# 记录所有图片路径
all_images = []
for class_name in classes:
    class_folder = os.path.join(dataset_path, 'train', class_name)
    images = [os.path.join(class_name, image) for image in os.listdir(class_folder)]
    all_images += images

# 随机打乱图片路径
random.shuffle(all_images)

# 划分训练和验证数据集
train_images = all_images[:int(len(all_images) * 0.8)]
val_images = all_images[int(len(all_images) * 0.8):]

# 写入train.txt
with open(train_path, 'w') as f:
    for image in train_images:
        class_name = image.split('/')[0]
        f.write('{} {}\n'.format(image, classes_dict[class_name]))

# 写入val.txt
with open(val_path, 'w') as f:
    for image in val_images:
        class_name = image.split('/')[0]
        f.write('{} {}\n'.format(image, classes_dict[class_name]))
# # 写入train.txt
# with open(train_path, 'w') as f:
#     for image_path, label in train_images:
#         f.write(f'{image_path} {label}\n')
#
# with open(val_path, 'w') as f:
#     for image_path, label in val_images:
#         f.write(f'{image_path} {label}\n')