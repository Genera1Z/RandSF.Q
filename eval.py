"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl

import cv2
import numpy as np
import torch as pt
import tqdm

from object_centric_bench.datum import DataLoader
from object_centric_bench.util_datum import draw_segmentation_np
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.util import Config, build_from_config


@pt.inference_mode()
def val_epoch(
    cfg,
    dataset_v,
    model,
    loss_fn_v,
    acc_fn_v,
    callback_v,
    is_viz=False,
    is_img=False,
    dump_log=False,
):
    pack = Config({})
    pack.dataset_v = dataset_v
    pack.model = model
    pack.loss_fn_v = loss_fn_v
    pack.acc_fn_v = acc_fn_v
    pack.callback_v = callback_v
    pack.epoch = 0

    pack2 = Config({})

    mean = pt.from_numpy(np.array(cfg.IMAGENET_MEAN, "float32"))
    std = pt.from_numpy(np.array(cfg.IMAGENET_STD, "float32"))
    cnt = 0

    pack.model.eval()
    pack.isval = True
    [_.before_epoch(**pack) for _ in pack.callback_v]

    for i, batch in enumerate(tqdm.tqdm(pack.dataset_v)):
        pack.batch = batch

        [_.before_step(**pack) for _ in pack.callback_v]

        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(**pack)
            [_.after_forward(**pack) for _ in pack.callback_v]
            pack.loss = pack.loss_fn_v(**pack)
        pack.acc = pack.acc_fn_v(**pack)

        if is_viz:
            # mkdir
            save_dn = Path(cfg.name)
            if not Path(save_dn).exists():
                save_dn.mkdir(exist_ok=True)
            # read gt image and segment
            img_key = "image" if is_img else "video"
            imgs_gt = (  # image video
                (pack.batch[img_key] * std.cuda() + mean.cuda()).clip(0, 255).byte()
            )
            segs_gt = pack.batch["segment"]
            # read pd attent -> pd segment
            segs_pd = pack.output["segment"]
            # visualize gt image or video
            for img_gt, seg_gt, seg_pd in zip(imgs_gt, segs_gt, segs_pd):
                if is_img:
                    img_gt, seg_gt, seg_pd = [  # warp img as vid
                        _[None] for _ in (img_gt, seg_gt, seg_pd)
                    ]
                for tcnt, (igt, sgt, spd) in enumerate(zip(img_gt, seg_gt, seg_pd)):
                    igt = igt.permute(1, 2, 0).cpu().numpy()
                    igt = cv2.cvtColor(igt, cv2.COLOR_RGB2BGR)
                    sgt = sgt.cpu().numpy()
                    spd = spd.cpu().numpy()
                    save_path = save_dn / f"{cnt:06d}-{tcnt:06d}"
                    cv2.imwrite(f"{save_path}-i.png", igt)
                    cv2.imwrite(
                        f"{save_path}-s.png", draw_segmentation_np(igt, sgt, alpha=0.9)
                    )
                    cv2.imwrite(
                        f"{save_path}-p.png", draw_segmentation_np(igt, spd, alpha=0.9)
                    )
                cnt += 1

        [_.after_step(**pack) for _ in pack.callback_v]

    [_.after_epoch(**pack) for _ in pack.callback_v]

    for cb in pack.callback_v:
        flag_log = False
        if cb.__class__.__name__ == "AverageLog":
            flag_log = True
            pack2.log_info = cb.mean()
        elif cb.__class__.__name__ == "HandleLog":
            flag_log = True
            pack2.log_info = cb.handle()
        if flag_log:
            if dump_log:
                with open(f"{cfg.name}.pkl", "wb") as f:
                    pkl.dump(cb.state_dict, f)
            break

    return pack2


def main(args):
    pt.backends.cudnn.benchmark = True

    assert args.cfg_file.name.endswith(".py")
    assert args.cfg_file.is_file()
    cfg_name = args.cfg_file.name.split(".")[0]
    cfg = Config.fromfile(args.cfg_file)
    cfg.name = cfg_name

    ## datum init

    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = args.data_dir

    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        cfg.batch_size_v,
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
    )

    ## model init

    model = build_from_config(cfg.model)
    # print(model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)

    if args.ckpt_file:
        model.load(args.ckpt_file, None, verbose=False)
    if cfg.freez:
        model.freez(cfg.freez, verbose=False)

    model = model.cuda()
    # model.compile()

    ## learn init

    loss_fn_v = MetricWrap(**build_from_config(cfg.loss_fn_v))
    acc_fn_v = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))

    cfg.callback_v = [_ for _ in cfg.callback_v if _.type.__name__ != "SaveModel"]
    for cb in cfg.callback_v:
        if cb.type.__name__ in ["AverageLog", "HandleLog"]:
            cb.log_file = None
    callback_v = build_from_config(cfg.callback_v)

    ## do eval

    pack2 = val_epoch(
        cfg,
        dataload_v,
        model,
        loss_fn_v,
        acc_fn_v,
        callback_v,
        args.is_viz,
        args.is_img,
        args.dump_log,
    )

    return pack2.log_info


def main_eval_multi(args):
    cfg_files = [
        # "config-randsfq/randsfq_c-movi_c.py",
        # "config-randsfq/randsfq_c-movi_e.py",
        # "config-randsfq/randsfq_r-ytvis.py",
        # "config-randsfq/randsfq_r-ytvis_hq.py",
        # "config-randsfq-tsim/randsfq_c-movi_c-tsim.py",
        # "config-randsfq-tsim/randsfq_c-movi_e-tsim.py",
        "config-randsfq-tsim/randsfq_r-ytvis-tsim.py",
        # "config-randsfq-tsim/randsfq_r-ytvis_hq-tsim.py",
        # "config-slotcontrast/slotcontrast_c-movi_c.py",
        # "config-slotcontrast/slotcontrast_c-movi_e.py",
        "config-slotcontrast/slotcontrast_r-ytvis.py",
        # "config-slotcontrast/slotcontrast_r-ytvis_hq.py",
        # "config-videosaur/videosaur_c-movi_c.py",
        # "config-videosaur/videosaur_c-movi_e.py",
        "config-videosaur/videosaur_r-ytvis.py",
        # "config-videosaur/videosaur_r-ytvis_hq.py",
        #
        # "config-randsfq/randsfq_r_recogn-ytvis_hq.py",
        # "config-slotcontrast/slotcontrast_r_recogn-ytvis_hq.py",
    ]
    ckpt_files = [
        # [
        #     "archive-randsfq/randsfq_c-movi_c/42-0034.pth",
        #     "archive-randsfq/randsfq_c-movi_c/43-0036.pth",
        #     "archive-randsfq/randsfq_c-movi_c/44-0029.pth",
        # ],
        # [
        #     "archive-randsfq/randsfq_c-movi_e/42-0037.pth",
        #     "archive-randsfq/randsfq_c-movi_e/43-0035.pth",
        #     "archive-randsfq/randsfq_c-movi_e/44-0032.pth",
        # ],
        # [
        #     "archive-randsfq/randsfq_r-ytvis/42-0121.pth",
        #     "archive-randsfq/randsfq_r-ytvis/43-0117.pth",
        #     "archive-randsfq/randsfq_r-ytvis/44-0124.pth",
        # ],
        # [
        #     "archive-randsfq/randsfq_r-ytvis_hq/42-0120.pth",
        #     "archive-randsfq/randsfq_r-ytvis_hq/43-0146.pth",
        #     "archive-randsfq/randsfq_r-ytvis_hq/44-0120.pth",
        # ],
        # [
        #     "archive-randsfq-tsim/randsfq_c-movi_c/42-0031.pth",
        #     "archive-randsfq-tsim/randsfq_c-movi_c/43-0029.pth",
        #     "archive-randsfq-tsim/randsfq_c-movi_c/44-0032.pth",
        # ],
        # [
        #     "archive-randsfq-tsim/randsfq_c-movi_e/42-0041.pth",
        #     "archive-randsfq-tsim/randsfq_c-movi_e/43-0030.pth",
        #     "archive-randsfq-tsim/randsfq_c-movi_e/44-0030.pth",
        # ],
        [
            "archive-randsfq-tsim/randsfq_r-ytvis-tsim/42-0111.pth",
            "archive-randsfq-tsim/randsfq_r-ytvis-tsim/43-0085.pth",
            "archive-randsfq-tsim/randsfq_r-ytvis-tsim/44-0088.pth",
        ],
        # [
        #     "archive-randsfq-tsim/randsfq_r-ytvis_hq/42-0155.pth",
        #     "archive-randsfq-tsim/randsfq_r-ytvis_hq/43-0120.pth",
        #     "archive-randsfq-tsim/randsfq_r-ytvis_hq/44-0172.pth",
        # ],
        # [
        #     "archive-slotcontrast/slotcontrast_c-movi_c/42-0034.pth",
        #     "archive-slotcontrast/slotcontrast_c-movi_c/43-0032.pth",
        #     "archive-slotcontrast/slotcontrast_c-movi_c/44-0033.pth",
        # ],
        # [
        #     "archive-slotcontrast/slotcontrast_c-movi_e/42-0028.pth",
        #     "archive-slotcontrast/slotcontrast_c-movi_e/43-0030.pth",
        #     "archive-slotcontrast/slotcontrast_c-movi_e/44-0033.pth",
        # ],
        [
            "archive-slotcontrast/slotcontrast_r-ytvis/42-0111.pth",
            "archive-slotcontrast/slotcontrast_r-ytvis/43-0101.pth",
            "archive-slotcontrast/slotcontrast_r-ytvis/44-0130.pth",
        ],
        # [
        #     "archive-slotcontrast/slotcontrast_r-ytvis_hq/42-0155.pth",
        #     "archive-slotcontrast/slotcontrast_r-ytvis_hq/43-0155.pth",
        #     "archive-slotcontrast/slotcontrast_r-ytvis_hq/44-0172.pth",
        # ],
        # [
        #     "archive-videosaur/videosaur_c-movi_c/42-0034.pth",
        #     "archive-videosaur/videosaur_c-movi_c/43-0041.pth",
        #     "archive-videosaur/videosaur_c-movi_c/44-0028.pth",
        # ],
        # [
        #     "archive-videosaur/videosaur_c-movi_e/42-0025.pth",
        #     "archive-videosaur/videosaur_c-movi_e/43-0027.pth",
        #     "archive-videosaur/videosaur_c-movi_e/44-0026.pth",
        # ],
        [
            "archive-videosaur/videosaur_r-ytvis/42-0091.pth",
            "archive-videosaur/videosaur_r-ytvis/43-0104.pth",
            "archive-videosaur/videosaur_r-ytvis/44-0104.pth",
        ],
        # [
        #     "archive-videosaur/videosaur_r-ytvis_hq/42-0112.pth",
        #     "archive-videosaur/videosaur_r-ytvis_hq/43-0155.pth",
        #     "archive-videosaur/videosaur_r-ytvis_hq/44-0120.pth",
        # ],
        # [
        #     "archive-recogn/randsfq_r_recogn-ytvis_hq/42-0023.pth",
        #     "archive-recogn/randsfq_r_recogn-ytvis_hq/43-0022.pth",
        #     "archive-recogn/randsfq_r_recogn-ytvis_hq/44-0022.pth",
        # ],
        # [
        #     "archive-recogn/slotcontrast_r_recogn-ytvis_hq/42-0022.pth",
        #     "archive-recogn/slotcontrast_r_recogn-ytvis_hq/43-0020.pth",
        #     "archive-recogn/slotcontrast_r_recogn-ytvis_hq/44-0020.pth",
        # ],
    ]
    ckpt_base_dir = Path("/media/GeneralZ/Storage/Active/0_ckpt_randsfq_github")

    assert len(cfg_files) == len(ckpt_files)

    log_file = Path("eval_multi.csv")
    log_file.touch()
    keys = ("ari", "ari_fg", "mbo", "miou")
    # keys = ("top1", "top3", "iou", "num")
    for cfgf, ckptfs in zip(cfg_files, ckpt_files):
        cfgf = Path(cfgf)
        for ckptf in ckptfs:
            ckptf = ckpt_base_dir / ckptf
            cname = ckptf.parent.name
            assert cname == cfgf.name[:-3]
            seed = int(ckptf.name.split("-")[0])
            print(f"###\n{cname}-{seed}\n###")
            print(cfgf.as_posix(), ckptf.as_posix())
            args.cfg_file = cfgf
            args.ckpt_file = ckptf
            eval_info = main(args)
            values = [eval_info[_] for _ in keys]
            values_str = ",".join([f"{_:.8f}" for _ in values])
            with open(log_file, "a") as f:
                f.writelines(f"{cname}-{seed},{values_str}\n")
    return


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg_file",
        type=Path,  # TODO XXX
        default="config-smoothsa/smoothsa_r-coco.py",
    )
    parser.add_argument(  # TODO XXX
        "--data_dir", type=Path, default="/media/GeneralZ/Storage/Static/datasets"
    )
    parser.add_argument(
        "--ckpt_file",
        type=Path,  # TODO XXX
        default="archive-smoothsa/smoothsa_r-coco/42-0027.pth",
    )
    parser.add_argument(
        "--is_viz",
        type=bool,  # TODO XXX
        default=False,
    )
    parser.add_argument(
        "--is_img",  # image or video
        type=bool,  # TODO XXX
        default=False,
    )
    parser.add_argument(
        "--dump_log",
        type=bool,  # TODO XXX
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
    # main_eval_multi(parse_args())
