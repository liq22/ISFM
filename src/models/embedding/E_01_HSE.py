import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class E_01_HSE(nn.Module):
    """
    Randomly divides the input tensor along L (length) and C (channel) dimensions into patches,
    then mixes these patches using linear layers. After the patches are selected, a time embedding
    is added based on the sampling period, ensuring temporal information is preserved without
    including the time axis in patch selection.

    Args:
        patch_size_L (int): Patch size along the L dimension.
        patch_size_C (int): Patch size along the C dimension.
        num_patches (int): Number of random patches to sample.
        output_dim (int): Output feature dimension after linear mixing.
        f_s (int): Sampling frequency, used to compute sampling period (T = 1/f_s).
    """
    def __init__(self, args,args_d):
        super(E_01_HSE, self).__init__()
        self.patch_size_L = args.patch_size_L  # Patch size along L dimension
        self.patch_size_C = args.patch_size_C  # Patch size along C dimension
        self.num_patches = args.n_patches    # Number of patches to sample
        self.output_dim =  args.output_dim
        self.args_d = args_d   
        # self.f_s =  args_d.f_s  # Sampling frequency
        # self.T = 1.0 /  args_d.f_s  # Sampling period


        # Two linear layers for flatten + mixing
        self.linear1 = nn.Linear(self.patch_size_L * (self.patch_size_C * 2), self.output_dim)
        self.linear2 = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x: torch.Tensor,data_name) -> torch.Tensor:
        """
        Forward pass of RandomPatchMixer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C),
                              where B is batch size, L is length, C is channels.

        Returns:
            torch.Tensor: Output tensor of shape (B, num_patches, output_dim).
        """
        B, L, C = x.size()
        device = x.device
        fs = self.args_d.task[data_name]['f_s']
        T = 1.0 / fs

        # Generate time axis 't' for each sample, shape: (B, L)
        t = torch.arange(L, device=device, dtype=torch.float32) * T
        t = t.unsqueeze(0).expand(B, -1)

        # If input is smaller than required patch size, repeat along L or C as needed
        if self.patch_size_L > L:
            repeats_L = (self.patch_size_L + L - 1) // L  # Ceiling division
            x = repeat(x, 'b l c -> b (l r) c', r=repeats_L)
            t = repeat(t, 'b l -> b (l r)', r=repeats_L)
            L = x.size(1)

        if self.patch_size_C > C:
            repeats_C = (self.patch_size_C + C - 1) // C
            x = repeat(x, 'b l c -> b l (c r)', r=repeats_C)
            C = x.size(2)

        # Randomly sample starting positions for patches
        max_start_L = L - self.patch_size_L
        max_start_C = C - self.patch_size_C
        start_indices_L = torch.randint(0, max_start_L + 1, (B, self.num_patches), device=device)
        start_indices_C = torch.randint(0, max_start_C + 1, (B, self.num_patches), device=device)

        # Create offsets for patch sizes
        offsets_L = torch.arange(self.patch_size_L, device=device)
        offsets_C = torch.arange(self.patch_size_C, device=device)

        # Compute actual indices
        idx_L = (start_indices_L.unsqueeze(-1) + offsets_L) % L  # (B, num_patches, patch_size_L)
        idx_C = (start_indices_C.unsqueeze(-1) + offsets_C) % C  # (B, num_patches, patch_size_C)

        # Expand for advanced indexing
        idx_L = idx_L.unsqueeze(-1)  # (B, num_patches, patch_size_L, 1)
        idx_C = idx_C.unsqueeze(-2)  # (B, num_patches, 1, patch_size_C)

        # Gather patches
        patches = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)
        patches = patches.gather(2, idx_L.expand(-1, -1, -1, C))
        patches = patches.gather(3, idx_C.expand(-1, -1, self.patch_size_L, -1))

        # Gather corresponding time embeddings
        t_expanded = t.unsqueeze(1).expand(-1, self.num_patches, -1)  # (B, num_patches, L)
        t_patches = t_expanded.gather(2, idx_L.squeeze(-1))           # (B, num_patches, patch_size_L)
        t_patches = t_patches.unsqueeze(-1).expand(-1, -1, -1, self.patch_size_C)

        # Concatenate time embedding to the end along channel dimension
        patches = torch.cat([patches, t_patches], dim=-1)  # shape: (B, num_patches, patch_size_L, patch_size_C + 1)

        # Flatten each patch and apply linear layers
        patches = rearrange(patches, 'b p l c -> b p (l c)')
        out = self.linear1(patches)
        out = F.silu(out)
        out = self.linear2(out)
        return out
if __name__ == '__main__':
    # Testing the RandomPatchMixer class
    def test_random_patch_mixer():
        B = 2  # Batch size
        L_list = [1024, 2048]  # Variable sequence lengths
        C_list = [8, 3]   # Variable channel dimensions

        patch_size_L = 128   # Patch size along L dimension
        patch_size_C = 5   # Patch size along C dimension
        num_patches = 100   # Number of patches to sample
        output_dim = 16    # Output dimension after mixing
        f_s = 100  # Sampling frequency

        model = E_01_HSE(patch_size_L, patch_size_C, num_patches, output_dim, f_s)

        for C in C_list:
            for L in L_list:
                x = torch.randn(B, L, C)
                y = model(x)
                print(f'Input shape: {x.shape}, Output shape: {y.shape}')

    # Run the test
    test_random_patch_mixer()
