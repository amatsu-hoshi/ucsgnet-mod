import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import sdf
import wandb, hydra
from model import CSGNet

@hydra.main(version_base=None, config_path="./config", config_name="config")
def run(config):
    wandb.init(project='ucsgnet', name=config.misc.name)
    model = train(config)


def make_sdf_data(tree_structure) -> sdf.SDF3: 
    if getattr(tree_structure, "leftop", None) is not None: 
        return getattr(sdf, tree_structure.operator)(make_sdf_data(tree_structure.leftop), make_sdf_data(tree_structure.rightop))
    else: 
        f = getattr(sdf, tree_structure.shape_type)(tree_structure.params)
        f = f.translate(tree_structure.translation)
        return f

def sample(f, N_points, bbox_min, bbox_max, balanced): 
    bbox_min = numpy.array(bbox_min) - 0.01
    bbox_max = numpy.array(bbox_max) + 0.01
    if not balanced: 
        points = numpy.random.uniform(low=bbox_min, high=bbox_max, size=(N_points, 3)).astype(numpy.float32)
        occupancies = (f(points) < 0).astype(numpy.float32)[:, 0]
    else: 
        # sample equal points inside and outside the shape
        inside_points = numpy.empty((0, 3), dtype=numpy.float32)
        outside_points = numpy.empty((0, 3), dtype=numpy.float32)
        while inside_points.shape[0] < N_points // 2 or outside_points.shape[0] < N_points // 2:
            sampled_points = numpy.random.uniform(
                low=bbox_min, high=bbox_max, size=(N_points, 3)).astype(numpy.float32)
            sdf_values = f(sampled_points)[:, 0]
            if inside_points.shape[0] < N_points // 2:
                is_inside = sdf_values < 0
                inside_points = numpy.concatenate(
                    (inside_points, sampled_points[is_inside]), axis=0)
            if outside_points.shape[0] < N_points // 2:
                is_outside = sdf_values >= 0
                outside_points = numpy.concatenate(
                    (outside_points, sampled_points[is_outside]), axis=0)
        inside_points = inside_points[:N_points // 2]
        outside_points = outside_points[:N_points // 2]
        points = numpy.concatenate((inside_points, outside_points), axis=0)
        occupancies = numpy.zeros(4096, dtype=numpy.float32)
        occupancies[:N_points // 2] = 1
    return torch.tensor(points).unsqueeze(0), torch.tensor(occupancies).unsqueeze(0)

def train(config):
    model = CSGNet(config.model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training.lr)
    step = 0
    # generate sdf function from config
    sdf_function = make_sdf_data(config.model.shape)
    bbox_min, bbox_max = sdf.mesh._estimate_bounds(sdf_function)
    for step in range(config.training.max_steps): 
        # sample new points from bounding box
        points, occupancies = sample(sdf_function, config.model.N_points, bbox_min, bbox_max, balanced=config.model.balanced)
        points = points.cuda()
        occupancies = occupancies.cuda()
        loss, loss_terms = model(points, occupancies)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        wandb.log({ 'step': step, **loss_terms })
        step += 1
    return model

if __name__ == '__main__': 
    run()