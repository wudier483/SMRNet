import os
import numpy as np
from utils.pose_gen import pose_generator
from utils.visualization import render_animation


def demo_visualize(mode, cfg, model, diffusion, dataset):
    """
    script for drawing gifs in different modes
    """
    if cfg.dataset != 'h36m' and mode != 'pred':
        raise NotImplementedError(f"sorry, {mode} is currently only available in h36m setting.")

    if mode == 'pred':
        action_list = dataset['test'].prepare_iter_action(cfg.dataset)
        for i in range(0, len(action_list)):
            pose_gen = pose_generator(dataset['test'], model, diffusion, cfg,
                                      mode='pred', action=action_list[i], nrow=cfg.vis_row)
            suffix = action_list[i]
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col + 2,
                             output=os.path.join(cfg.gif_dir, f'pred_{suffix}.gif'), mode=mode)
    else:
        raise
