from fastai.vision.all import *
from fastai.vision import *
import os
import torch
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def Train(cfg):
    df_train = pd.read_csv(cfg.CSV_PATH)  # placeholder
    df_train["image_name"] = df_train["image_name"].apply(
        lambda x: os.path.join(cfg.DATA_DIR, x)
    )
    tfms = aug_transforms(mult=0.37, p_affine=0.21)

    device = cfg.DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dls = all.ImageDataLoaders.from_df(
        df_train,
        label_col="label",
        bs=cfg.BATCH_SIZE,
        valid_col="is_valid",
        seed=cfg.SEED,
        item_tfms=Resize(
            cfg.IMG_SZ, method="squish", pad_mode="zeros", resamples=(0, 0)
        ),
        batch_tfms=tfms,
        device=torch.device(device),
    )

    precision = Precision(average="macro")
    recall = Recall(average="macro")

    all_lbls = df_train.label.tolist()
    cls_wts = compute_class_weight(
        class_weight="balanced", classes=np.unique(all_lbls), y=all_lbls
    )

    if device == "cuda":
        cls_wts_tensor = FloatTensor(cls_wts).cuda()
    elif device == "cpu":
        cls_wts_tensor = FloatTensor(cls_wts)

    # can change opt_func
    learn = vision_learner(
        dls,
        cfg.MODELNAME,
        pretrained=True,
        loss_func=LabelSmoothingCrossEntropy(weight=cls_wts_tensor),
        metrics=[error_rate, accuracy, precision, recall],
        opt_func=RAdam,
    )

    # change fname path with colab title name

    cb1 = SaveModelCallback(
        monitor="train_accuracy",
        min_delta=0.0,
        fname="models/Violeta_M_2_vit_base_patch32_224_in21k",
        every_epoch=False,
        at_end=False,
        with_opt=False,
        reset_on_fit=True,
    )
    # cb2 = EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=20)
    # cb3 = ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=2)
    cb4 = ShowGraphCallback()
    mixup = MixUp(alpha=0.17)
    cb5 = Recorder(train_metrics=True)
    learn.fine_tune(epochs=cfg.EPOCHS, freeze_epochs=5, cbs=[cb1, mixup, cb4, cb5])
    learn.export(cfg.EXPORT_PATH)
