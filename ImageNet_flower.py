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

# ????????????????????????????????????????????? PyTorch ???????????????????????????????????????????????? PyTorch ??????????????????????????????
optimizer = dict(type='SGD',  # ???????????????
                 lr=0.1,  # ??????????????????????????????????????????????????????????????? PyTorch ?????????
                 momentum=0.9,  # ??????(Momentum)
                 weight_decay=0.0001)  # ??????????????????(weight decay)???
# optimizer hook ???????????????
optimizer_config = dict(grad_clip=None)  # ????????????????????????????????????(grad_clip)???
# ???????????????????????????????????? LrUpdater hook???
lr_config = dict(policy='step',  # ????????????(scheduler)????????????????????? CosineAnnealing, Cyclic, ??????
                 step=[30, 60, 90])  # ??? epoch ??? 30, 60, 90 ?????? lr ????????????
runner = dict(type='EpochBasedRunner',  # ???????????? runner ??????????????? IterBasedRunner ??? EpochBasedRunner???
              max_epochs=100)  # runner ??????????????? ?????? IterBasedRunner ?????? `max_iters`

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
