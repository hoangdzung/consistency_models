"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import lpips
from tqdm import tqdm

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample


def updated_rotate_latents(latents, angle, rand):
    ori_shape = rand.shape
    latents = latents.reshape(latents.shape[0], -1)
    rand = rand.reshape(rand.shape[0], -1)
    
    angle = th.tensor(np.radians(angle))
    perpend_vectors = rand - (rand * latents).sum(dim=1, keepdim=True) * latents / (latents * latents).sum(dim=1, keepdim=True)
    perpend_vectors = perpend_vectors / th.norm(perpend_vectors, dim=1, keepdim=True) * th.norm(latents, dim=1, keepdim=True)
    new_latents = th.cos(angle) * latents + th.sin(angle) * perpend_vectors
    
    new_latents = new_latents.reshape(*ori_shape)
    return new_latents 

# Function to compute LPIPS loss between two images
def compute_lpips_loss(lpips_model, img1, img2):
    with th.no_grad():
        return lpips_model(img1, img2)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_labels, all_latents, all_images = [], [], []
    all_bad_labels, all_bad_latents, all_bad_images = [], [], []
    generator = get_generator(args.generator, args.num_samples, args.seed)
    lpips_model = lpips.LPIPS(net='vgg')

    args.batch_size = 1 
    with tqdm(total=args.num_samples) as pbar:
        while len(all_latents) < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes.repeat(args.n_tests + 1)

            x_T = generator.randn(args.batch_size, 3, args.image_size, args.image_size, device=dist_util.dev()) * args.sigma_max
            rand = generator.randn(args.n_tests, 3, args.image_size, args.image_size, device=dist_util.dev()) 
            neigh_x_T = updated_rotate_latents(x_T, args.angle, rand)
            all_x_T = th.cat([x_T, neigh_x_T])
            all_sample = karras_sample(
                diffusion,
                model,
                (args.n_tests + 1, 3, args.image_size, args.image_size),
                steps=args.steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                clip_denoised=args.clip_denoised,
                sampler=args.sampler,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                s_churn=args.s_churn,
                s_tmin=args.s_tmin,
                s_tmax=args.s_tmax,
                s_noise=args.s_noise,
                generator=generator,
                ts=ts,
                x_T=all_x_T
            )
            all_sample = ((all_sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            all_sample = all_sample.contiguous()
            distances = compute_lpips_loss(lpips_model.cuda(),all_sample[:1],all_sample[1:]) 
            if distances.max().item() <= args.threshold:
                all_latents.append(x_T.cpu().numpy())
                all_labels.append(classes.cpu().numpy())
                logger.log(f"created {len(all_latents) * args.batch_size} samples")
                pbar.update(1)  # Update the progress bar

    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels)
    # save file
    np.savez(os.path.join(args.save_dir, 'priors.npz'), x_T=all_latents, classes=all_labels)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
        save_dir="",
        n_tests=200,
        angle=1.0,
        threshold=0.01,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()


