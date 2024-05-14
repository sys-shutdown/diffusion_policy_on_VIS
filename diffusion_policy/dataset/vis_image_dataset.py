from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class VISImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['image1', 'image2', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image1'] = get_image_range_normalizer()
        normalizer['image2'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        
        image1 = np.moveaxis(sample['image1'],-1,1)/255
        image2 = np.moveaxis(sample['image2'],-1,1)/255

        data = {
            'obs': {
                'image1': image1, # T, 3, 400, 200
                'image2': image2, # T, 3, 400, 200
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import random
    import os
    import cv2
    zarr_path = os.path.expanduser('../TrainData/vis_demo.zarr')
    dataset = VISImageDataset(zarr_path, horizon=16)
    for j in range(200):
        data = dataset.__getitem__(random.randint(0,dataset.__len__()-1))
        print(data['action'])
        for i in range(len(data['obs']['image1'])):
            img = np.concatenate([np.moveaxis(data['obs']['image1'][i].numpy(),0,-1),np.moveaxis(data['obs']['image2'][i].numpy(),0,-1)],axis=1)
            cv2.imshow('1',img)
            cv2.waitKey(10)
    print("End")
    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

if __name__ == "__main__":
    test()
    