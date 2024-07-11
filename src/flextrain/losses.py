from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import (
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
)

from .types import Batch, TorchTensorN, TorchTensorNX


@dataclass
class LossOutput:
    metrics: Dict[str, torch.Tensor] = field(default_factory=dict)
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)


loss_function_type = Callable[[Batch, Any], LossOutput]


class LossSupervised(nn.Module):
    def __init__(
        self,
        fn_name: str,
        fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_key: str,
        weight: float = 1.0,
        post_process_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        fn_target: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        convert_target_to_output_type: bool = False,
    ):
        """
        parameters:
            fn: a functor expecting (predicted, target) inputs and returning the loss per sample
        """
        super().__init__()
        self.weight = weight
        self.fn_name = fn_name
        self.fn = fn
        self.batch_key = batch_key
        self.post_process_fn = post_process_fn
        self.convert_target_to_output_type = convert_target_to_output_type
        self.fn_target = fn_target

    def forward(self, batch: Batch, model_output: torch.Tensor, **kwargs: Any) -> LossOutput:
        output_target = batch[self.batch_key]
        if self.fn_target is not None:
            output_target = self.fn_target(output_target)

        if self.convert_target_to_output_type:
            if isinstance(model_output, (list, tuple)):
                # deep supervision: the output is actually a sequence
                output_target = output_target.type(model_output[0].dtype)
            else:
                output_target = output_target.type(model_output.dtype)

        loss = self.fn(model_output, output_target)

        if self.post_process_fn is not None:
            # for example if torchmetrics is used, convert the metric to loss
            loss = self.post_process_fn(loss)
        return LossOutput({}, {self.fn_name: loss * self.weight})


def squeeze_target_int(t: torch.Tensor) -> torch.Tensor:
    assert t.shape[1] == 1, f'got={t.shape}'
    assert t.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    return t[:, 0].type(torch.long)


# combines sigmoid layer with binary cross entropy
LossBceLogitsSigmoid = partial(
    LossSupervised, fn_name='bce', fn=nn.BCEWithLogitsLoss(reduction='none'), convert_target_to_output_type=True
)

LossCELogitsSoftmax = partial(
    LossSupervised,
    fn_name='ce',
    fn=nn.CrossEntropyLoss(reduction='none'),
    convert_target_to_output_type=False,
    fn_target=squeeze_target_int,
)

LossBce = partial(LossSupervised, fn_name='bce', fn=nn.BCELoss(reduction='none'), convert_target_to_output_type=True)

LossL1 = partial(LossSupervised, fn_name='l1', fn=nn.L1Loss(reduction='none'), convert_target_to_output_type=True)


class LossPerceptual(nn.Module):
    def __init__(
        self,
        perceptual_model: nn.Module,
        batch_key: str,
        input_transform_fn: Callable[[torch.Tensor], torch.Tensor] = torch.nn.Identity(),
        weight: float = 0.1,
    ) -> None:
        super().__init__()

        # make sure we don't update the perceptual model
        self.perceptual_model = perceptual_model
        for param in self.perceptual_model.parameters():
            param.requires_grad = False

        self.weight = weight
        self.batch_key = batch_key
        self.input_transform_fn = input_transform_fn

    def forward(self, batch: Batch, model_output: torch.Tensor, **kwargs: Any) -> LossOutput:
        metrics: Dict[str, torch.Tensor] = {}
        losses = {}

        with torch.no_grad():
            # no need for gradient in this branch: this is static!
            features_real = self.perceptual_model(self.input_transform_fn(batch[self.batch_key]))
        features_recon = self.perceptual_model(self.input_transform_fn(model_output))

        perceptual_loss = nn.functional.l1_loss(features_recon, features_real, reduction='none')
        losses['perceptual'] = perceptual_loss.view((perceptual_loss.shape[0], -1)).mean(dim=1) * self.weight
        return LossOutput(metrics, losses)


class LossCombine(nn.Module):
    def __init__(self, **losses: Any) -> None:
        super().__init__()
        self.losses = nn.ModuleDict(losses)

    def forward(self, batch: Batch, model_output: torch.Tensor, **kwargs: Any) -> LossOutput:
        metrics_all = {}
        losses_all = {}
        for loss_name, loss_fn in self.losses.items():
            loss_outputs = loss_fn(batch, model_output, **kwargs)

            for metric_name, metric in loss_outputs.metrics.items():
                name = f'{loss_name}_{metric_name}' if len(metric_name) > 0 else f'{loss_name}'
                metrics_all[name] = metric
            for l_name, loss in loss_outputs.losses.items():
                name = f'{loss_name}_{l_name}' if len(l_name) > 0 else f'{loss_name}'
                losses_all[name] = loss

        return LossOutput(metrics_all, losses_all)


class _LossTorchMetrics(nn.Module):
    def __init__(
        self,
        metric_fn: Any,
        batch_key: str,
        fn_name: str,
        weight: float = 1.0,
        input_preprocessing_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        loss_postprocessing_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: 1.0 - x,
        **torchmetric_kwargs: Any,
    ) -> None:
        super().__init__()

        # store the metric so that `.to(device)` works as expected with the lambda functions
        self.metric_fn = metric_fn(**torchmetric_kwargs)

        if isinstance(self.metric_fn, nn.Module):
            # metric parameters should NOT be modified!
            for param in self.metric_fn.parameters():
                param.requires_grad = False

        self.loss_fn = LossSupervised(
            fn_name=fn_name,
            fn=lambda output, target: self.metric_fn(input_preprocessing_fn(output), input_preprocessing_fn(target)),
            batch_key=batch_key,
            weight=weight,
            post_process_fn=loss_postprocessing_fn,
        )

    def forward(self, batch: Batch, model_output: torch.Tensor, **kwargs: Any) -> LossOutput:
        loss: LossOutput = self.loss_fn(batch, model_output)
        return loss


class LossSSIM(_LossTorchMetrics):
    def __init__(
        self,
        batch_key: str,
        data_range: Tuple[float, float],
        weight: float = 1,
        input_preprocessing_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        **torchmetric_kwargs: Any,
    ) -> None:
        assert isinstance(data_range, tuple), 'range must be specified as a tuple (min, max) values'
        assert len(data_range) == 2, 'min/max values!'

        super().__init__(
            # specify the data_range, else issues with noisy background!
            metric_fn=partial(StructuralSimilarityIndexMeasure, data_range=data_range),
            batch_key=batch_key,
            fn_name='ssim',
            weight=weight,
            input_preprocessing_fn=input_preprocessing_fn,
            loss_postprocessing_fn=lambda x: 1 - x,
            reduction='none',
            **torchmetric_kwargs,
        )

    def forward(self, batch: Batch, model_output: torch.Tensor, **kwargs: Any) -> LossOutput:
        # min_value = model_output.min()
        # if min_value < 0:
        # see https://www.researchgate.net/publication/
        # 358308582_On_the_proper_use_of_structural_similarity_for_the_robust_evaluation_of_medical_image_synthesis_models
        #    warnings.warn(f'LossSSIM should be used only with values >= 0. Got={min_value}')
        return super().forward(batch, model_output)


class LossMSSSIM(_LossTorchMetrics):
    def __init__(
        self,
        batch_key: str,
        data_range: Tuple[float, float],
        weight: float = 1,
        input_preprocessing_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        **torchmetric_kwargs: Any,
    ) -> None:
        assert isinstance(data_range, tuple), 'range must be specified as a tuple (min, max) values'
        assert len(data_range) == 2, 'min/max values!'

        super().__init__(
            # specify the data_range, else issues with noisy background!
            metric_fn=partial(MultiScaleStructuralSimilarityIndexMeasure, data_range=data_range),
            batch_key=batch_key,
            fn_name='msssim',
            weight=weight,
            input_preprocessing_fn=input_preprocessing_fn,
            loss_postprocessing_fn=lambda x: 1 - x,
            reduction='none',
            **torchmetric_kwargs,
        )

    def forward(self, batch: Batch, model_output: torch.Tensor, **kwargs: Any) -> LossOutput:
        # min_value = model_output.min()
        # if min_value < 0:
        # see https://www.researchgate.net/publication/
        # 358308582_On_the_proper_use_of_structural_similarity_for_the_robust_evaluation_of_medical_image_synthesis_models
        #    warnings.warn(f'LossSSIM should be used only with values >= 0. Got={min_value}')
        return super().forward(batch, model_output)


def preprocessing_rgb_m11(
    x: torch.Tensor,
    channel_mode: Literal['central', 'first', 'expand'] = 'expand',
) -> torch.Tensor:
    """
    Preprocessing of the input. Torchmetrics LPIPS are expected to have shape [N, 3, H, W] so process the
    input to satisfy this requirement

    parameters:
        channel_mode: `central` only the middle slice is used and this is repeated x3 (i.e., grayscale).
            `first` take the first 3 slices. `expand`: moving window of 3 components, the input is expanded

    Expecting input values in [-1, 1]

    Designed to be used with `_LossTorchMetrics`
    """
    assert len(x.shape) == 4, 'expecting NCHW format'
    assert x.min() >= (-1 - 1e-3), f'expected min >= -1, got={x.min()}'
    assert x.max() <= (1 + 1e-3), f'expected max >= -1, got={x.max()}'

    if channel_mode == 'central':
        if x.shape[1] > 1 and x.shape[1] != 3:
            # only consider the central slice
            central = x.shape[1] // 2
            x = x[:, central : central + 1]
            assert x.shape[1] == 1, 'expecting a single channel'
            return torch.concatenate([x] * 3, dim=1)

    if channel_mode == 'first':
        if x.shape[1] > 1 and x.shape[1] != 3:
            # only consider the central slice
            assert x.shape[1] > 3
            x = x[:, 0:3]
            return x

    if channel_mode == 'expand':
        if x.shape[1] > 1 and x.shape[1] != 3:
            # split the channels in groups of `3`
            assert x.shape[1] > 3
            xs = [x[:, i : i + 3] for i in range(0, x.shape[1] - 2, 1)]
            x = torch.concatenate(xs, 0)
            return x

    assert x.shape[1] == 3
    return x


LossLPIPS = partial(
    _LossTorchMetrics,
    metric_fn=LearnedPerceptualImagePatchSimilarity,
    fn_name='lpips',
    input_preprocessing_fn=preprocessing_rgb_m11,
    loss_postprocessing_fn=lambda x: x,  # low = good, high = bad (as opposed to what the `metric` implies...)
)


def contrastive_loss(x1: TorchTensorNX, x2: TorchTensorNX, same_class: TorchTensorN, margin: float = 1.0) -> TorchTensorNX:
    """
    Computes the contrastive Loss [1]

    [1] Learning a Similarity Metric Discriminatively, with Application to Face Verification,
        Sumit Chopra, Raia Hadsell and Yann LeCun
        http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf

    Args:
        x1: the first embedding
        x2: the second embedding
        same_class: a tensor where `1` means (x1, x2) are from the same class and `0` are from different classes

    Returns:
        a loss per embedding
    """

    dist = torch.nn.functional.pairwise_distance(x1, x2)
    assert same_class.max() <= 1  # 0 = dissimilar, 1 = similar
    assert same_class.shape == (len(x1),)

    loss: torch.Tensor = same_class * dist**2 + torch.clamp(margin - dist, min=0.0) ** 2
    return loss


class LossContrastive(nn.Module):
    def __init__(self, same_class_fn: Callable[[Sequence[Batch]], torch.Tensor]) -> None:
        super().__init__()
        self.same_class_fn = same_class_fn

    def forward(
        self, positive_negative_batches: Sequence[Batch], positive_negative_model_outputs: Sequence[torch.Tensor]
    ) -> LossOutput:
        assert len(positive_negative_batches) == len(positive_negative_model_outputs)
        assert len(positive_negative_batches) == 2, 'Use only 2 positive and negative pairs'
        same_class = self.same_class_fn(positive_negative_batches)
        constrative_loss_value = contrastive_loss(
            positive_negative_model_outputs[0], positive_negative_model_outputs[1], same_class
        )
        return LossOutput({}, {'contrastive': constrative_loss_value})


class LossRanking(nn.Module):
    def __init__(self, true_ordering_fn: Callable[[Batch, Batch], torch.Tensor], margin: float = 0.05) -> None:
        super().__init__()
        self.true_ordering_fn = true_ordering_fn
        self.margin = margin

    def forward(self, batches: Sequence[Batch], outputs: Sequence[torch.Tensor], **kwargs: Any) -> LossOutput:
        assert len(outputs) == 2
        assert len(batches) == 2
        true_ordering = self.true_ordering_fn(batches[0], batches[1])
        assert true_ordering.max() <= 1 and true_ordering.min() >= 0, 'must be binary!'
        true_ordering_1m_1 = 2 * (1 - true_ordering.float()) - 1  # `1` means batch[0] should be higher than batch[1]
        loss = torch.nn.MarginRankingLoss(reduction='none', margin=self.margin)(outputs[0], outputs[1], true_ordering_1m_1)
        output = outputs[0] < outputs[1]
        accuracy = (output == true_ordering).sum() / float(len(true_ordering_1m_1))
        return LossOutput({'1-accuracy': 1.0 - accuracy}, {'rank': loss})


class LossCrossEntropy(nn.Module):
    """
    Calculate the cross entropy between logit (multiclass) and a target (ordinal)
    """

    def __init__(
        self,
        target_name: str,
        weight: float = 1.0,
        class_weights: Optional[Union[Literal['adaptative'], Tuple[float, ...]]] = None,
        weight_min_max_adaptative: Tuple[Optional[float], Optional[float]] = (None, None),
        reduction: str = 'none',
    ):
        super().__init__()
        self.target_name = target_name
        self.weight = weight
        if isinstance(class_weights, str):
            self.class_weights = class_weights
            assert class_weights in ('adaptative',)
        else:
            self.class_weights = torch.asarray(class_weights) if class_weights is not None else None

        if weight_min_max_adaptative[0] is not None or weight_min_max_adaptative[1] is not None:
            self.weight_min_max_adaptative = weight_min_max_adaptative
        else:
            self.weight_min_max_adaptative = None
        self.reduction = reduction

    def forward(self, batch: Batch, model_output: torch.Tensor, **kwargs: Any) -> LossOutput:
        targets = batch[self.target_name]
        assert len(model_output) == len(targets)
        if len(model_output.shape) == len(targets.shape):
            # For an input NC[D]HW, pytorch expects to have N x [D]HW as target
            # In this framework, target is N1[D]HW
            assert targets.shape[1] == 1
            assert model_output.shape[2:] == targets.shape[2:]
            targets = targets.squeeze(1)

        if targets.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            targets = targets.type(torch.int64)

        w = None
        if self.class_weights is not None:
            if self.class_weights == 'adaptative':
                nb_classes = torch.bincount(torch.flatten(targets), minlength=model_output.shape[1])
                nb_voxels = torch.prod(torch.asarray(targets.shape), dtype=torch.int64)
                w = (nb_voxels / nb_classes) / 5

                if self.weight_min_max_adaptative is not None:
                    w = torch.clip(w, self.weight_min_max_adaptative[0], self.weight_min_max_adaptative[1])

            else:
                w = self.class_weights.to(model_output.device)
            assert len(w) == model_output.shape[1]
        loss = nn.functional.cross_entropy(model_output, targets, reduction=self.reduction, weight=w)
        accuracy = (model_output.argmax(dim=1) == targets).sum().float().cpu() / targets.nelement()
        return LossOutput({'1-accuracy': 1.0 - accuracy}, {'ce': loss * self.weight})
