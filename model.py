import typing as t

import torch
import torch.nn as nn

from common import RNN_LATENT_SIZE, TrainingStage
from csg_layers import RelationLayer, Scaler
from shape_evaluators import CompundEvaluator, PlanesEvaluator, create_compound_evaluator
from losses import *


def occupancy_operator(operator: str, leftop: torch.Tensor, rightop: torch.Tensor): 
    if operator == 'union': 
        return torch.clamp(leftop + rightop, min=0, max=1)
    elif operator == 'intersection': 
        return torch.clamp(leftop + rightop - 1, min=0, max=1)
    elif operator == 'difference': 
        return torch.clamp(leftop - rightop, min=0, max=1)

class CSGTree: 
    def __init__(self, tree_structure, number_of_shapes) -> None:
        if getattr(tree_structure, "leftop", None) is not None: 
            # Non-leaf nodes
            self.leftop = CSGTree(tree_structure.leftop, number_of_shapes) 
            self.rightop = CSGTree(tree_structure.rightop, number_of_shapes)
            self.operator = tree_structure.operator
        else: 
            # Basic shapes
            self.shape_type = tree_structure.shape_type
            if self.shape_type in number_of_shapes.keys(): 
                # index in its type
                self.id = number_of_shapes[self.shape_type]
                number_of_shapes[self.shape_type] += 1
            else: 
                self.id = 0
                number_of_shapes[self.shape_type] = 1

    def compute_id(self, number_of_shapes: dict): 
        if getattr(self, "shape_type", None) is None:
            self.leftop.compute_id(number_of_shapes)
            self.rightop.compute_id(number_of_shapes)
        else: 
            for shape_type, num_shapes in number_of_shapes.items(): 
                if shape_type != self.shape_type: 
                    self.id += num_shapes
                else:
                    break

    def evaluate(self, base_shapes): 
        if getattr(self, "shape_type", None) is None: 
            return occupancy_operator(self.operator, self.leftop.evaluate(base_shapes), self.rightop.evaluate(base_shapes))
        else: 
            return base_shapes[..., self.id]

class CSGNet(nn.Module):
    def __init__(
        self, 
        hparams
    ):
        super().__init__()
        number_of_shapes = {}
        self.csg_tree = CSGTree(hparams.shape, number_of_shapes)
        # map nodes in csg_tree to indices base_shapes, must call this in before evaluation
        self.csg_tree.compute_id(number_of_shapes)
        self.evaluator_ = create_compound_evaluator(number_of_shapes, 3)
        self.scaler_ = Scaler()
        self.w_translation_loss = hparams.w_translation_loss
        self.w_recon_loss = hparams.w_recon_loss
        self.w_scaling_loss = hparams.w_scaling_loss
        self.w_positive_parameter_loss = hparams.w_positive_parameter_loss

    def forward(
        self,
        points: torch.Tensor, 
        occupancies: torch.Tensor
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, ...]]:
        points = points.unsqueeze(
            dim=1
        )  # broadcasting for different of shapes

        base_shapes = self.evaluator_(
            points
        )  # -> batch, num_shapes, num_points

        base_shapes = base_shapes.permute(
            (0, 2, 1)
        )  # -> batch, num_points, num_shapes

        scaled_shapes = 1 - self.scaler_(base_shapes)
        occupancies_pred = self.csg_tree.evaluate(scaled_shapes)
        occupancies_pred = torch.clamp(occupancies_pred, min=0, max=1)
        translation_loss = get_translation_loss(self.evaluator_)
        positive_parameter_loss = get_positive_parameter_loss(self.evaluator_)
        scaling_loss = get_scaling_loss(self.scaler_)
        recon_loss = get_recon_loss(occupancies_pred, occupancies)
        loss = (
            translation_loss * self.w_translation_loss + 
            scaling_loss * self.w_scaling_loss + 
            recon_loss * self.w_recon_loss + 
            positive_parameter_loss * self.w_positive_parameter_loss
        )
        loss_terms = {
            'translation_loss': translation_loss, 
            'scaling_loss': scaling_loss, 
            'positive_parameter_loss': positive_parameter_loss, 
            'recon_loss': recon_loss, 
            'loss': loss
        }
        return loss, loss_terms