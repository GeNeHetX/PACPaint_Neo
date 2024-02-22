import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List

class MLP(torch.nn.Sequential):
    """
    MLP Module

    Parameters
    ----------
    in_features: int
    out_features: int
    hidden: Optional[List[int]] = None
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        d_model = in_features
        layers = []

        if hidden is not None:
            for i, h in enumerate(hidden):
                seq = [torch.nn.Linear(d_model, h, bias=bias)]
                d_model = h

                if activation is not None:
                    seq.append(activation)

                layers.append(torch.nn.Sequential(*seq))

        layers.append(torch.nn.Linear(d_model, out_features))

        super(MLP, self).__init__(*layers)


def get_neo_preds(
    slidename: str,
    PATH_NEO: Path,
    PATH_TEMP_DIR: Path,
    device: torch.device = torch.device("cuda:0"),
    PATH_FEATURES: Path = None,
    features=None,
) -> pd.DataFrame:
    assert PATH_FEATURES is not None or features is not None, "Either PATH_FEATURES or features must be provided"

    if features is None:
        features = np.load(PATH_FEATURES, mmap_mode="r")
    coord = np.array(features[:, :3])
    x = np.array(features[:, 3:])

    model = MLP(
        in_features=x.shape[-1],
        out_features=1,
        hidden=[128],
        activation=torch.nn.ReLU(),
    )
    model.load_state_dict(torch.load(PATH_NEO,map_location="cpu"))
    # Add a sigmoid layer to the model
    model.sigmoid = torch.nn.Sigmoid()
    model.to(device)
    model.eval()

    x_torch = torch.tensor(x).float().to(device)

    with torch.inference_mode():
        preds_ = model(x_torch)
        preds_ = preds_.cpu().numpy().squeeze()
        preds_ = pd.DataFrame({"z": coord[:, 0], "x": coord[:, 1], "y": coord[:, 2], "pred": preds_})

    preds_.to_csv(PATH_TEMP_DIR / f"{slidename}" / "preds_neo.csv", index=False)

    return preds_
