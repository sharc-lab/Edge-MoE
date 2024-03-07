#!/usr/bin/env python3

from pathlib import Path
import numpy as np

INPUT_DIRECTORY = Path(__file__).parent.parent
OUTPUT_DIRECTORY = INPUT_DIRECTORY / 'onboard'
NUM_LAYERS = 12
NUM_TASKS = 2

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

np.save(
    OUTPUT_DIRECTORY / 'patch_embed_weights.npy',
    np.fromfile(
        INPUT_DIRECTORY / 'patch_embed_weight.float32.bin',
        dtype=np.float32,
    ).reshape((192, 3, 16, 16))
)
np.save(
    OUTPUT_DIRECTORY / 'patch_embed_bias.npy',
    np.fromfile(
        INPUT_DIRECTORY / 'patch_embed_bias.float32.bin',
        dtype=np.float32,
    ).reshape((192,))
)
np.save(
    OUTPUT_DIRECTORY / 'pos_embed.npy',
    np.fromfile(
        INPUT_DIRECTORY / 'pos_embed.float32.bin',
        dtype=np.float32,
    ).reshape((129, 192))
)

np.save(
    OUTPUT_DIRECTORY / 'norm_weights.npy',
    np.stack([np.stack([np.fromfile(
                            INPUT_DIRECTORY / f'l{l}_norm{n}_weight.float32.bin',
                            dtype=np.float32,
                        ).reshape((192,))
                        for n in range(1, 3)]) for l in range(NUM_LAYERS)])
)
np.save(
    OUTPUT_DIRECTORY / 'norm_bias.npy',
    np.stack([np.stack([np.fromfile(
                            INPUT_DIRECTORY / f'l{l}_norm{n}_bias.float32.bin',
                            dtype=np.float32,
                        ).reshape((192,))
                        for n in range(1, 3)]) for l in range(NUM_LAYERS)])
)

np.save(
    OUTPUT_DIRECTORY / 'attn_weights.npy',
    np.stack([np.concatenate((
        np.stack(np.split(np.fromfile(
            INPUT_DIRECTORY / f'l{l}_qkv_weight.float32.bin',
            dtype=np.float32,
        ).reshape((576, 192)), 3)),
        np.fromfile(
            INPUT_DIRECTORY / f'l{l}_attn_proj_weight.float32.bin',
            dtype=np.float32,
        ).reshape((1, 192, 192)),
    )) for l in range(NUM_LAYERS)])
)
np.save(
    OUTPUT_DIRECTORY / 'attn_bias.npy',
    np.stack([np.concatenate((
        np.stack(np.split(np.fromfile(
            INPUT_DIRECTORY / f'l{l}_qkv_bias.float32.bin',
            dtype=np.float32,
        ).reshape((576,)), 3)),
        np.fromfile(
            INPUT_DIRECTORY / f'l{l}_attn_proj_bias.float32.bin',
            dtype=np.float32,
        ).reshape((1, 192)),
    )) for l in range(NUM_LAYERS)])
)

np.save(
    OUTPUT_DIRECTORY / 'vit_weights_l1.npy',
    np.stack([np.fromfile(
                  INPUT_DIRECTORY / f'l{l}_fc1_weight.float32.bin',
                  dtype=np.float32,
              ).reshape((768, 192))
              for l in range(0, NUM_LAYERS, 2)])
)
np.save(
    OUTPUT_DIRECTORY / 'vit_bias_l1.npy',
    np.stack([np.fromfile(
                  INPUT_DIRECTORY / f'l{l}_fc1_bias.float32.bin',
                  dtype=np.float32,
              ).reshape((768,))
              for l in range(0, NUM_LAYERS, 2)])
)
np.save(
    OUTPUT_DIRECTORY / 'vit_weights_l2.npy',
    np.stack([np.fromfile(
                  INPUT_DIRECTORY / f'l{l}_fc2_weight.float32.bin',
                  dtype=np.float32,
              ).reshape((192, 768))
              for l in range(0, NUM_LAYERS, 2)])
)
np.save(
    OUTPUT_DIRECTORY / 'vit_bias_l2.npy',
    np.stack([np.fromfile(
                  INPUT_DIRECTORY / f'l{l}_fc2_bias.float32.bin',
                  dtype=np.float32,
              ).reshape((192,))
              for l in range(0, NUM_LAYERS, 2)])
)

np.save(
    OUTPUT_DIRECTORY / 'moe_weights_l1.npy',
    np.stack([np.fromfile(
                  INPUT_DIRECTORY / f'l{l}_htoh4_weight.float32.bin',
                  dtype=np.float32,
              ).reshape((16, 384, 192))
              for l in range(1, NUM_LAYERS, 2)])
)
np.save(
    OUTPUT_DIRECTORY / 'moe_bias_l1.npy',
    np.stack([np.fromfile(
                  INPUT_DIRECTORY / f'l{l}_htoh4_bias.float32.bin',
                  dtype=np.float32,
              ).reshape((16, 384))
              for l in range(1, NUM_LAYERS, 2)])
)
np.save(
    OUTPUT_DIRECTORY / 'moe_weights_l2.npy',
    np.stack([np.fromfile(
                  INPUT_DIRECTORY / f'l{l}_h4toh_weight.float32.bin',
                  dtype=np.float32,
              ).reshape((16, 192, 384))
              for l in range(1, NUM_LAYERS, 2)])
)
np.save(
    OUTPUT_DIRECTORY / 'moe_bias_l2.npy',
    np.stack([np.fromfile(
                  INPUT_DIRECTORY / f'l{l}_h4toh_bias.float32.bin',
                  dtype=np.float32,
              ).reshape((16, 192))
              for l in range(1, NUM_LAYERS, 2)])
)

np.save(
    OUTPUT_DIRECTORY / 'moe_w_gate_per_task.npy',
    np.stack([np.stack([np.fromfile(
                            INPUT_DIRECTORY / f'l{l}_w_gate_T_task{task}.float32.bin',
                            dtype=np.float32,
                        ).reshape((16, 192))
                        for l in range(1, NUM_LAYERS, 2)]) for task in range(NUM_TASKS)])
)
