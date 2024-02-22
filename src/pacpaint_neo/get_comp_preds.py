import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Union

import warnings


class MaskedLinear(torch.nn.Linear):
    """
    Linear layer to be applied tile wise.
    This layer can be used in combination with a mask
    to prevent padding tiles from influencing the values of a subsequent
    activation.
    Example:
        >>> module = Linear(in_features=128, out_features=1) # With Linear
        >>> out = module(slide)
        >>> wrong_value = torch.sigmoid(out) # Value is influenced by padding
        >>> module = MaskedLinear(in_features=128, out_features=1, mask_value='-inf') # With MaskedLinear
        >>> out = module(slide, mask) # Padding now has the '-inf' value
        >>> correct_value = torch.sigmoid(out) # Value is not influenced by padding as sigmoid('-inf') = 0
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    mask_value: Union[str, int]
        value to give to the mask
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_value: Union[str, float],
        bias: bool = True,
    ):
        super(MaskedLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        x: torch.Tensor
            (B, SEQ_LEN, OUT_FEATURES)
        """
        x = super(MaskedLinear, self).forward(x)
        if mask is not None:
            x = x.masked_fill(mask, float(self.mask_value))
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}, mask_value={}, bias={}".format(
            self.in_features, self.out_features, self.mask_value, self.bias is not None
        )


class TilesMLP(torch.nn.Module):
    """
    MLP to be applied to tiles to compute scores.
    This module can be used in combination of a mask
    to prevent padding from influencing the scores values.
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    hidden: Optional[List[int]] = None:
        Number of hidden layers and their respective number of features.
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    activation: torch.nn.Module = torch.nn.Sigmoid()
        MLP activation function
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden: Optional[List[int]] = None,
        bias: bool = True,
        activation: torch.nn.Module = torch.nn.Sigmoid(),
    ):
        super(TilesMLP, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        if hidden is not None:
            for h in hidden:
                self.hidden_layers.append(MaskedLinear(in_features, h, bias=bias, mask_value="-inf"))
                self.hidden_layers.append(activation)
                in_features = h

        self.hidden_layers.append(torch.nn.Linear(in_features, out_features, bias=bias))

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES), True for values that were padded.
        Returns
        -------
        x: torch.Tensor
            (B, N_TILES, OUT_FEATURES)
        """
        for layer in self.hidden_layers:
            if isinstance(layer, MaskedLinear):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x


class ExtremeLayer(torch.nn.Module):
    """
    Extreme layer.
    Returns concatenation of n_top top tiles and n_bottom bottom tiles

    .. warning::
        If top tiles or bottom tiles is superior to the true number of tiles in the input then padded tiles will
        be selected and their value will be 0.

    Parameters
    ----------
    n_top: int
        number of top tiles to select
    n_bottom: int
        number of bottom tiles to select
    dim: int
        dimension to select top/bottom tiles from
    return_indices: bool
        Whether to return the indices of the extreme tiles
    """

    def __init__(
        self,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        dim: int = 1,
        return_indices: bool = False,
    ):
        super(ExtremeLayer, self).__init__()

        if not (n_top is not None or n_bottom is not None):
            raise ValueError("one of n_top or n_bottom must have a value.")

        if not ((n_top is not None and n_top > 0) or (n_bottom is not None and n_bottom > 0)):
            raise ValueError("one of n_top or n_bottom must have a value > 0.")

        self.n_top = n_top
        self.n_bottom = n_bottom
        self.dim = dim
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, ...)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, ...)

        Warnings
        --------
        If top tiles or bottom tiles is superior to the true number of tiles in the input then padded tiles will
        be selected and their value will be 0.

        Returns
        -------
        extreme_tiles: torch.Tensor
            (B, N_TOP + N_BOTTOM, ...)
        """

        if self.n_top and self.n_bottom and ((self.n_top + self.n_bottom) > x.shape[self.dim]):
            warnings.warn(
                f"Sum of tops is larger than the input tensor shape for dimension {self.dim}: "
                + f"{self.n_top + self.n_bottom} > {x.shape[self.dim]}. Values will appear twice (in top and in bottom)"
            )

        top, bottom = None, None
        top_idx, bottom_idx = None, None
        if mask is not None:
            if self.n_top:
                top, top_idx = x.masked_fill(mask, float("-inf")).topk(k=self.n_top, sorted=True, dim=self.dim)
                top_mask = top.eq(float("-inf"))
                if top_mask.any():
                    warnings.warn("The top tiles contain masked values, they will be set to zero.")
                    top[top_mask] = 0

            if self.n_bottom:
                bottom, bottom_idx = x.masked_fill(mask, float("inf")).topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )
                bottom_mask = bottom.eq(float("inf"))
                if bottom_mask.any():
                    warnings.warn("The bottom tiles contain masked values, they will be set to zero.")
                    bottom[bottom_mask] = 0
        else:
            if self.n_top:
                top, top_idx = x.topk(k=self.n_top, sorted=True, dim=self.dim)
            if self.n_bottom:
                bottom, bottom_idx = x.topk(k=self.n_bottom, largest=False, sorted=True, dim=self.dim)

        if top is not None and bottom is not None:
            values = torch.cat([top, bottom], dim=self.dim)
            indices = torch.cat([top_idx, bottom_idx], dim=self.dim)
        elif top is not None:
            values = top
            indices = top_idx
        elif bottom is not None:
            values = bottom
            indices = bottom_idx
        else:
            raise ValueError

        if self.return_indices:
            return values, indices
        else:
            return values

    def extra_repr(self):
        return f"n_top={self.n_top}, n_bottom={self.n_bottom}"


class Weldon(torch.nn.Module):
    """
    Weldon module.

    Parameters
    ----------
    in_features: int
    out_features: int
        controls the number of scores and, by extension, the number of out_features
    tiles_mlp_hidden: Optional[List[int]] = None
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        n_extreme: Optional[int] = 10,
        tiles_mlp_hidden: Optional[List[int]] = None,
        bias: bool = True,
    ):
        super(Weldon, self).__init__()

        self.score_model = TilesMLP(in_features, hidden=tiles_mlp_hidden, bias=bias, out_features=out_features)
        self.extreme_layer = ExtremeLayer(n_top=n_extreme, n_bottom=n_extreme)


class WeldonInference(Weldon):
    def __init__(self, in_features, out_features, n_extreme, tiles_mlp_hidden):
        super(WeldonInference, self).__init__(
            in_features=in_features,
            out_features=out_features,
            n_extreme=n_extreme,
            tiles_mlp_hidden=tiles_mlp_hidden,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES), Beware, N_TILES must be bigger than n_extreme.
            Also, the model is expected to be less good if N_TILES < 2*n_extreme. A warning will be issued.
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits, extreme_scores: Tuple[torch.Tensor, torch.Tensor]:
            (B, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES)
        """
        scores = self.score_model(x=x, mask=mask)
        try:
            extreme_scores = self.extreme_layer(x=scores, mask=mask)
        except RuntimeError as e:
            print(f"Error in extreme layer with {x.shape=} and {scores.shape=}. Verify that N_TILES > n_extreme.")
            raise e

        score_wsi = torch.mean(extreme_scores, 1, keepdim=False)
        return score_wsi, scores


def get_comp_preds(
    slidename: str,
    PATH_COMP: Path,
    PATH_TEMP_DIR: Path,
    device: torch.device = torch.device("cuda:0"),
    PATH_FEATURES: Path = None,
    features=None,
    PATH_COL_NAMES: Path = None,
):
    assert PATH_FEATURES is not None or features is not None, "Either PATH_FEATURES or features must be provided"

    # col_names = np.loadtxt(
    #     PATH_COL_NAMES,
    #     dtype=str,
    # )
    col_names = [
        "Classic",
        "Basal",
    ]
    if features is None:
        features = np.load(PATH_FEATURES, mmap_mode="r")
    coord = np.array(features[:, :3])
    x = np.array(features[:, 3:])

    model = WeldonInference(
        in_features=x.shape[1],
        out_features=len(col_names),
        n_extreme=100,
        tiles_mlp_hidden=[128],
    )
    model.load_state_dict(torch.load(PATH_COMP))
    # model = model.score_model
    model.to(device)
    model.eval()

    x_torch = torch.tensor(x).float().to(device)

    df_pred_tiles = pd.DataFrame(coord[:, [0, 1, 2]], columns=["z", "x", "y"])
    with torch.inference_mode():
        score_wsi, scores_tiles = model.forward(x_torch.unsqueeze(0))
        score_wsi = score_wsi.cpu().detach().numpy()
        scores_tiles = scores_tiles.cpu().numpy().squeeze()

        df_pred_wsi = pd.DataFrame(score_wsi, columns=col_names)
        df_pred_tiles_ = pd.DataFrame(scores_tiles, columns=col_names)
        df_pred_tiles = pd.concat([df_pred_tiles, df_pred_tiles_], axis=1)

        df_pred_wsi.to_csv(PATH_TEMP_DIR / f"{slidename}" / "preds_comp_wsi.csv", index=False)
        df_pred_tiles.to_csv(PATH_TEMP_DIR / f"{slidename}" / "preds_comp_tiles.csv", index=False)

        # preds_ = model(x_torch)
        # preds_ = preds_.cpu().numpy().squeeze()
        # df_pred_ = pd.DataFrame(preds_, columns=col_names)
        # df_pred = pd.concat([df_pred, df_pred_], axis=1)
        # df_pred.to_csv(PATH_TEMP_DIR / f"{slidename}" / "preds_comp.csv", index=False)

    return df_pred_wsi, df_pred_tiles
