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
from PIL import Image

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
    ori_shape_latents = latents.shape  # (10, 1, 3, 64, 64)
    ori_shape_rand = rand.shape        # (10, N, 3, 64, 64)
    
    # Reshape latents to 10x1x(3*64*64)
    latents = latents.reshape(ori_shape_latents[0], ori_shape_latents[1], -1)  # (10, 1, 3*64*64)
    
    # Reshape rand to 10xNx(3*64*64)
    rand = rand.reshape(ori_shape_rand[0], ori_shape_rand[1], -1)  # (10, N, 3*64*64)
    
    angle_rad = th.tensor(np.radians(angle), dtype=th.float32)
    
    # Compute dot products
    dot_product = (rand * latents).sum(dim=2, keepdim=True)  # (10, N, 1)
    norm_latents = (latents * latents).sum(dim=2, keepdim=True)  # (10, 1, 1)
    
    # Compute perpendicular vectors
    perpend_vectors = rand - (dot_product * latents) / norm_latents  # (10, N, 3*64*64)
    norm_perpend_vectors = th.norm(perpend_vectors, dim=2, keepdim=True)  # (10, N, 1)
    norm_latents_unsqueezed = th.norm(latents, dim=2, keepdim=True)  # (10, 1, 1)
    perpend_vectors = perpend_vectors / norm_perpend_vectors * norm_latents_unsqueezed  # (10, N, 3*64*64)
    
    # Compute new latents
    new_latents = th.cos(angle_rad) * latents + th.sin(angle_rad) * perpend_vectors  # (10, N, 3*64*64)
    
    # Reshape new_latents back to original shape 10xNx3x64x64
    new_latents = new_latents.reshape(ori_shape_rand)  # (10, N, 3, 64, 64)
    
    return new_latents


# Function to compute LPIPS loss between two images
def compute_lpips_loss(lpips_model, img1, img2):
    with th.no_grad():
        return lpips_model(img1, img2)

def check_angle_between_vectors(vec1, vec2, angle):
    # Flatten the tensors
    vec1_flat = vec1.reshape(-1)
    vec2_flat = vec2.reshape(-1)
    
    # Compute cosine similarity
    dot_product = (vec1_flat * vec2_flat).sum()
    norm_vec1 = th.norm(vec1_flat)
    norm_vec2 = th.norm(vec2_flat)
    
    cos_sim = dot_product / (norm_vec1 * norm_vec2)
    
    # Compare with cosine of the given angle
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    
    return cos_sim, th.isclose(cos_sim, th.tensor(cos_angle, dtype=th.float32), atol=1e-6)

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

    n_stable_samples = 0
    n_unstable_samples = 0
    n_total_samples = 0

    n_actual_point_per_batch = args.batch_size // (args.num_neighbors + 1)
    args.batch_size = (args.num_neighbors + 1) * n_actual_point_per_batch

    generator = get_generator(args.generator, args.num_samples, args.seed)
    lpips_model = lpips.LPIPS(net='vgg').to(dist_util.dev())
    with tqdm(total=args.num_samples) as pbar:
        while n_stable_samples < args.num_samples:
            model_kwargs = {"return_intermediate": args.return_intermediate}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(n_actual_point_per_batch,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes.repeat_interleave(args.num_neighbors + 1)

            x_T = generator.randn(n_actual_point_per_batch, 1, 3, args.image_size, args.image_size, device=dist_util.dev())
            rand = generator.randn(n_actual_point_per_batch, args.num_neighbors, 3, args.image_size, args.image_size, device=dist_util.dev()) 
            neigh_x_T = updated_rotate_latents(x_T, args.angle, rand)
            x_T *= args.sigma_max
            neigh_x_T *= args.sigma_max 
            all_x_T = th.cat([x_T, neigh_x_T], dim=1)
            all_x_T = all_x_T.reshape(-1, 3, args.image_size, args.image_size)

            all_sample = karras_sample(
                diffusion,
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
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
            if args.return_intermediate:
                all_sample, all_intermediates = all_sample

            all_sample = ((all_sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            all_sample = all_sample.contiguous()

            # to test the reshaping works, set all_sample = all_x_T and test cosine
            # all_sample = all_x_T

            # Reshape the batch to group images (n_actual_points , (num_neighbors + 1) , 3, 64, 64)
            grouped_batch = all_sample.view(n_actual_point_per_batch, (args.num_neighbors + 1) , 3, args.image_size, args.image_size)
            
            # Select the anchor images (first image in each group)
            anchor_images = grouped_batch[:, 0]  # Shape: (n_actual_points, 3, 64, 64)
            
            # Select the paired images (all other images in each group)
            neighbor_images = grouped_batch[:, 1:]  # Shape: (n_actual_points, num_neighbors, 3, 64, 64)
            
            # Repeat anchors to match the shape of pairs
            repeated_anchors = anchor_images.unsqueeze(1).expand(-1, args.num_neighbors, 3, args.image_size, args.image_size) # Shape: (n_actual_points, num_neighbors, 3, 64, 64)
            # Compute LPIPS loss for each pair
            
            lpips_distances = compute_lpips_loss(lpips_model, 
                                           repeated_anchors.reshape(-1, 3, args.image_size, args.image_size), 
                                           neighbor_images.reshape(-1, 3, args.image_size, args.image_size)) 
            
            lpips_distances_reshaped = lpips_distances.view(grouped_batch[:, 1:].shape[0], -1)
            max_lpips_distances, max_lpips_distance_indices = th.max(lpips_distances_reshaped, dim=1)
            max_distance_neighbors = neigh_x_T[th.arange(neigh_x_T.shape[0]), max_lpips_distance_indices]
            max_distance_neighbor_images = neighbor_images[th.arange(neighbor_images.shape[0]), max_lpips_distance_indices]

            l2_distances = th.sqrt(th.sum((repeated_anchors-neighbor_images)**2,dim=[2,3,4]))
            max_l2_distances = l2_distances[th.arange(l2_distances.shape[0]), max_lpips_distance_indices]

            if args.return_intermediate:
                intermediates_distances = {}
                for layer_idx, intermediates in enumerate(all_intermediates):
                    intermediate_shape = intermediates.shape
                    grouped_batch = intermediates.view(n_actual_point_per_batch, (args.num_neighbors + 1) , *intermediate_shape[-3:])
                    anchor_intermediates = grouped_batch[:, 0]
                    neighbor_intermediates = grouped_batch[:, 1:]
                    repeated_anchor_intermediates = anchor_intermediates.unsqueeze(1).expand(-1, args.num_neighbors, *intermediate_shape[-3:]) # Shape: (n_actual_points, num_neighbors, 3, 64, 64)
                    l2_distance = th.sqrt(th.sum((neighbor_intermediates - repeated_anchor_intermediates)**2, dim=[2, 3, 4]))
                    intermediates_distances[layer_idx] = l2_distance[th.arange(l2_distance.shape[0]), max_lpips_distance_indices]

            for i in range(n_actual_point_per_batch):
                max_distance = max_lpips_distances[i].item()
                if max_distance <= args.threshold:
                    n_stable_samples += 1
                    pbar.update(1)
                else:
                    n_unstable_samples += 1

                data = {
                    "x_T": x_T[i].cpu().numpy(),
                    "class": classes[i].cpu().numpy(),
                    "neigh_x_T": max_distance_neighbors[i].cpu().numpy(),
                    "sample": anchor_images[i].cpu().numpy(),
                    "neigh_sample": max_distance_neighbor_images[i].cpu().numpy(),
                    "lpips": max_lpips_distances[i].cpu().numpy(),
                    "l2": max_l2_distances[i].cpu().numpy(),
                }
                if args.return_intermediate:
                    data["intermediate_distances"] = np.array([intermediates_distances[layer_idx][i].cpu().numpy() for layer_idx in range(len(all_intermediates))])

                np.savez(os.path.join(args.save_dir, f'data_{n_total_samples:05d}.npz'), **data)
                n_total_samples += 1 
                if n_total_samples <= n_actual_point_per_batch:
                    # Convert images from numpy arrays to PIL images
                    sample_pil = Image.fromarray(np.transpose(data['sample'], (1, 2, 0)).astype(np.uint8))  # Transpose and convert to PIL image
                    neigh_sample_pil = Image.fromarray(np.transpose(data['neigh_sample'], (1, 2, 0)).astype(np.uint8))  # Transpose and convert to PIL image
                    
                    # Create a new composite image with samples side by side
                    composite_image = Image.new('RGB', (sample_pil.width * 2, sample_pil.height))
                    composite_image.paste(sample_pil, (0, 0))
                    composite_image.paste(neigh_sample_pil, (sample_pil.width, 0))
                    
                    # Save the composite image as PNG
                    composite_image.save(os.path.join(args.save_dir, f'data_{n_total_samples:05d}_{data["lpips"]:0.6f}.png'))
    
    print(n_stable_samples, n_unstable_samples)
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
        return_intermediate=True,
        num_neighbors=1,
        angle=1.0,
        threshold=0.01,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()


