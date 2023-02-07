# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/media/root/9E46A26D46A2463D/OpenMMLab/hw1/flower_dataset/train',
        ann_file='/media/root/9E46A26D46A2463D/OpenMMLab/hw1/flower_dataset/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/media/root/9E46A26D46A2463D/OpenMMLab/hw1/flower_dataset/val',
        ann_file='/media/root/9E46A26D46A2463D/OpenMMLab/hw1/flower_dataset/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/media/root/9E46A26D46A2463D/OpenMMLab/hw1/flower_dataset/val',
        ann_file='/media/root/9E46A26D46A2463D/OpenMMLab/hw1/flower_dataset/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['accuracy'])

# 用于构建优化器的配置文件。支持 PyTorch 中的所有优化器，同时它们的参数与 PyTorch 里的优化器参数一致。
optimizer = dict(type='SGD',  # 优化器类型
                 lr=0.1,  # 优化器的学习率，参数的使用细节请参照对应的 PyTorch 文档。
                 momentum=0.9,  # 动量(Momentum)
                 weight_decay=0.0001)  # 权重衰减系数(weight decay)。
# optimizer hook 的配置文件
optimizer_config = dict(grad_clip=None)  # 大多数方法不使用梯度限制(grad_clip)。
# 学习率调整配置，用于注册 LrUpdater hook。
lr_config = dict(policy='step',  # 调度流程(scheduler)的策略，也支持 CosineAnnealing, Cyclic, 等。
                 step=[30, 60, 90])  # 在 epoch 为 30, 60, 90 时， lr 进行衰减
runner = dict(type='EpochBasedRunner',  # 将使用的 runner 的类别，如 IterBasedRunner 或 EpochBasedRunner。
              max_epochs=100)  # runner 总回合数， 对于 IterBasedRunner 使用 `max_iters`

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/media/root/9E46A26D46A2463D/OpenMMLab/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = '/media/root/9E46A26D46A2463D/OpenMMLab/hw1/Res34'
gpu_ids = range(0, 1)
