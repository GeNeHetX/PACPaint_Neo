import os

# Demerdez-vous pour installer openslide sur votre machine
OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from display_wsi import display_wsi_results
from extract_tiles import filter_whites
from extract_features import extract_features
from get_neo_preds import get_neo_preds
from get_comp_preds import get_comp_preds

from pathlib import Path
from torch import device

from argparse import ArgumentParser

from torch import device


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument(
        "--temp_dir",
        type=Path,
        default=Path(r"D:\PACPaint_homemade\temp_folder"),
        help="Path to the temporary directory where the features will be saved",
        required=True,
    )
    parser.add_argument(
        "--wsi",
        type=Path,
        default=Path(r"D:\PACPaint_homemade\datasets\HES_PAC_MULTICENTRIC_Ambroise Pare_ok\B00127107-002.svs"),
        help="Path to the WSI. Can be a .svs, .ndpi, .qptiff",
    )
    parser.add_argument(
        "--neo",
        type=Path,
        default=Path(r"..\models\model_neo.pth"),
        help="Path to the neo model",
    )
    parser.add_argument(
        "--comp",
        type=Path,
        default=Path(r"..\models\model_comp_BASAL_CLASSIC_only.pth"),
        help="Path to the comp model",
    )
    parser.add_argument(
        "--device", type=device, default="cuda:0", help="Device to use for the predictions", choices=["cuda:0", "cpu"]
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for the feature extraction")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the feature extraction. Set to 0 if using windows.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="Prefetch factor for the feature extraction. Set to None if using windows.",
    )
    parser.add_argument("--pred_threshold", type=float, default=0.5, help="Threshold for the predictions")
    parser.add_argument("--display", action="store_true", help="Display the WSI and the tiles")

    parsed_args = parser.parse_args()
    if parsed_args.prefetch_factor is not None and parsed_args.num_workers == 0:
        raise ValueError("Prefetch factor can only be used when num_workers > 0")
    return parser.parse_args()


def main(args):
    slidename = args.wsi.stem
    print("Filtering white tiles...")
    tiles_coord = filter_whites(args.wsi, args.temp_dir)

    print("Extracting features...")
    features = extract_features(
        args.wsi,
        args.device,
        args.batch_size,
        tiles_coords_path=args.temp_dir / f"{slidename}" / "tiles_coord.npy",
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    PATH_FEATURES = args.temp_dir / f"{slidename}" / "features.npy"
    # np.save(PATH_FEATURES, features)

    print("Predicting neo...")
    pred_neo = get_neo_preds(slidename, args.neo, args.temp_dir, args.device, PATH_FEATURES=PATH_FEATURES)

    # print("Predicting comp...")
    # pred_comp_wsi, pred_comp = get_comp_preds(
    #     slidename, args.comp, args.temp_dir, args.device, PATH_FEATURES=PATH_FEATURES
    # )

    print("Done")

    if args.display:
        display_wsi_results(
            path_wsi=args.wsi,
            pred_neo=pred_neo,
            # pred_comp=pred_comp,
            thresh=args.pred_threshold,
        )


if __name__ == "__main__":
    args = parse_arg()
    main(args)
