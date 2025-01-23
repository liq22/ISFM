import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class RandomPatchMixer(nn.Module):
    """
    Randomly divides the input tensor along L and C dimensions into patches and mixes them using linear layers.
    Time embedding is added to the patches after they have been selected, preserving the temporal information
    without including the time axis in the patch selection process.
    """
    def __init__(self, patch_size_L, patch_size_C, num_patches, output_dim, f_s):
        super(RandomPatchMixer, self).__init__()
        self.patch_size_L = patch_size_L  # Patch size along L dimension
        self.patch_size_C = patch_size_C  # Patch size along C dimension
        self.num_patches = num_patches    # Number of patches to sample
        self.output_dim = output_dim
        self.f_s = f_s  # Sampling frequency
        self.T = 1.0 / f_s  # Sampling period

        # Linear layers for alignment and mixing
        # Input dimension remains the same as before, time embedding will be added after patch selection
        self.linear1 = nn.Linear(self.patch_size_L * (self.patch_size_C * 2), output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        B, L, C = x.size()
        device = x.device

        # Generate time axis t, starting from 0
        t = torch.arange(L, device=device).float() * self.T  # Shape: (L,)
        t = t.unsqueeze(0).expand(B, -1)  # Shape: (B, L)

        # Adjust x to fit the patch sizes if necessary by repeating elements
        if self.patch_size_L > L:
            repeats_L = (self.patch_size_L + L - 1) // L  # Ceiling division
            x = repeat(x, 'b l c -> b (l r) c', r=repeats_L)
            t = repeat(t, 'b l -> b (l r)', r=repeats_L)
            L = x.size(1)  # Update L

        if self.patch_size_C > C:
            repeats_C = (self.patch_size_C + C - 1) // C  # Ceiling division
            x = repeat(x, 'b l c -> b l (c r)', r=repeats_C)
            C = x.size(2)  # Update C

        # Randomly select starting positions for patches along L and C dimensions
        max_start_L = L - self.patch_size_L
        max_start_C = C - self.patch_size_C

        start_indices_L = torch.randint(0, max_start_L + 1, (B, self.num_patches), device=device)  # (B, num_patches)
        start_indices_C = torch.randint(0, max_start_C + 1, (B, self.num_patches), device=device)  # (B, num_patches)

        # Create offsets for patch sizes
        offsets_L = torch.arange(self.patch_size_L, device=device)  # (patch_size_L,)
        offsets_C = torch.arange(self.patch_size_C, device=device)  # (patch_size_C,)

        # Compute indices for patches
        idx_L = start_indices_L.unsqueeze(-1) + offsets_L  # (B, num_patches, patch_size_L)
        idx_C = start_indices_C.unsqueeze(-1) + offsets_C  # (B, num_patches, patch_size_C)

        # Use modulo operation to handle cases where indices exceed dimensions
        idx_L = idx_L % L  # (B, num_patches, patch_size_L)
        idx_C = idx_C % C  # (B, num_patches, patch_size_C)

        # Expand batch dimension for indexing
        idx_L = idx_L.unsqueeze(-1)  # (B, num_patches, patch_size_L, 1)
        idx_C = idx_C.unsqueeze(-2)  # (B, num_patches, 1, patch_size_C)

        # Extract patches using advanced indexing
        patches = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)  # (B, num_patches, L, C)
        patches = patches.gather(2, idx_L.expand(-1, -1, -1, C))  # (B, num_patches, patch_size_L, C)
        patches = patches.gather(3, idx_C.expand(-1, -1, self.patch_size_L, -1))  # (B, num_patches, patch_size_L, patch_size_C)

        # After patches are selected, add time embedding
        t_expanded = t.unsqueeze(1).expand(-1, self.num_patches, -1)  # (B, num_patches, L)
        t_patches = t_expanded.gather(2, idx_L.squeeze(-1))  # Extract corresponding time values, shape: (B, num_patches, patch_size_L)

        # Reshape t_patches to match patches
        t_patches = t_patches.unsqueeze(-1).expand(-1, -1, -1, self.patch_size_C)  # (B, num_patches, patch_size_L, patch_size_C)

        # Concatenate time embedding to the patches along the channel dimension
        patches = torch.cat([patches, t_patches], dim=-1)  # New patches shape: (B, num_patches, patch_size_L, patch_size_C + 1)

        # Flatten patches and apply linear layers
        patches = rearrange(patches, 'b p l c -> (b p) (l c)')  # (B * num_patches, patch_size_L * (patch_size_C + 1))
        out = self.linear1(patches)
        out = F.silu(out)
        out = self.linear2(out)

        # Reshape back to (B, num_patches, output_dim)
        out = rearrange(out, '(b p) o -> b p o', b=B, p=self.num_patches)

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

        model = RandomPatchMixer(patch_size_L, patch_size_C, num_patches, output_dim, f_s)

        for C in C_list:
            for L in L_list:
                x = torch.randn(B, L, C)
                y = model(x)
                print(f'Input shape: {x.shape}, Output shape: {y.shape}')

    # Run the test
    test_random_patch_mixer()
