import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class Patchfy(nn.Module):
    """
    Extracts patches from the input tensor and applies FFT along the L dimension.
    """
    def __init__(self, patch_size_L, patch_size_C, num_patches, f_s):
        super(Patchfy, self).__init__()
        self.patch_size_L = patch_size_L
        self.patch_size_C = patch_size_C
        self.num_patches = num_patches
        self.f_s = f_s
        self.T = 1.0 / f_s

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

        # Apply FFT along the L dimension
        patches_fft = torch.fft.fft(patches, dim=2)  # (B, num_patches, patch_size_L, patch_size_C)
        patches_fft = torch.view_as_real(patches_fft)  # (B, num_patches, patch_size_L, patch_size_C, 2)

        return patches_fft, t

class Fusion(nn.Module):
    """
    Fuses the patches using linear layers.
    """
    def __init__(self, patch_size_L, patch_size_C, output_dim):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(patch_size_L * patch_size_C * 2, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, patches_fft, t):
        B, num_patches, patch_size_L, patch_size_C, _ = patches_fft.size()

        # Flatten patches and apply linear layers
        patches = rearrange(patches_fft, 'b p l c r -> (b p) (l c r)')  # (B * num_patches, patch_size_L * patch_size_C * 2)
        out = self.linear1(patches)
        out = F.silu(out)
        out = self.linear2(out)

        # Reshape back to (B, num_patches, output_dim)
        out = rearrange(out, '(b p) o -> b p o', b=B, p=num_patches)

        return out

class RandomPatchMixer(nn.Module):
    """
    Randomly divides the input tensor along L and C dimensions into patches and mixes them using linear layers.
    Time embedding is added to the patches after they have been selected, preserving the temporal information
    without including the time axis in the patch selection process.
    """
    def __init__(self, patch_size_L, patch_size_C, num_patches, output_dim, f_s):
        super(RandomPatchMixer, self).__init__()
        self.patchfy = Patchfy(patch_size_L, patch_size_C, num_patches, f_s)
        self.fusion = Fusion(patch_size_L, patch_size_C, output_dim)

    def forward(self, x):
        patches_fft, t = self.patchfy(x)
        out = self.fusion(patches_fft, t)
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