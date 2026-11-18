from torchvision import transforms


IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    """
    Transformations used for the training set.

    Random augmentation is applied only during training to improve
    generalisation and reduce overfitting.
    """

    return transforms.Compose(
        [
            # 所有图片统一为 ResNet18 使用的 224 × 224
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

            # 模拟患者拍片时轻微的身体倾斜
            transforms.RandomRotation(
                degrees=7,
                fill=0,
            ),

            # 模拟轻微的位置、缩放和成像角度变化
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=3,
                fill=0,
            ),

            # 模拟不同设备和曝光条件造成的亮度、对比度差异
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),

            # 以较低概率模拟轻微模糊
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(0.1, 1.0),
                    )
                ],
                p=0.20,
            ),

            # 将 PIL 图片转换成 PyTorch Tensor
            transforms.ToTensor(),

            # 使用 ImageNet 预训练模型对应的标准化参数
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """
    Transformations used for validation and test sets.

    No random augmentation is used, so evaluation remains reproducible.
    """

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )