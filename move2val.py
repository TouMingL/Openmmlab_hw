import os
import shutil

# 源文件夹路径
src_folder = '/media/root/9E46A26D46A2463D/OpenMMLab/hw1/flower_dataset/train'
# 目标文件夹路径
dst_folder = '/media/root/9E46A26D46A2463D/OpenMMLab/hw1/flower_dataset/val'
# txt文件路径
txt_path = '/media/root/9E46A26D46A2463D/OpenMMLab/hw1/flower_dataset/val.txt'

# 读取txt文件
with open(txt_path, 'r') as f:
    lines = f.read().strip().split('\n')

# 遍历每行
for line in lines:
    src_path = os.path.join(src_folder, line.split(' ')[0])
    dst_path = os.path.join(dst_folder, line.split(' ')[0])

    # 如果源文件存在，则移动到目标文件夹
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
