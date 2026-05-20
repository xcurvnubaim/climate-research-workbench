"""
Script to modify sc2_resnet18_mae.ipynb:
1. Replace bilinear UpscaleBlock with PixelShuffle-based upscaling
2. Add LocallyConnected2D and ResidualLCB classes  
3. Integrate ResidualLCB into ResNet18SR model
"""

import json
import os

notebook_path = os.path.join(os.path.dirname(__file__), "sc2_resnet18_mae.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ============================================================
# 1. Find the cell containing UpscaleBlock and replace it
# ============================================================
upscale_cell_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "class UpscaleBlock" in source and "nn.Upsample" in source:
            upscale_cell_idx = i
            break

if upscale_cell_idx is None:
    print("ERROR: Could not find UpscaleBlock cell with nn.Upsample")
    exit(1)

print(f"Found UpscaleBlock cell at index {upscale_cell_idx}")

# The UpscaleBlock cell also contains DoubleConv, Down, and Up classes
# We need to preserve the parts before UpscaleBlock and replace the rest
old_source = nb["cells"][upscale_cell_idx]["source"]
old_source_str = "".join(old_source)

# Find where UpscaleBlock starts
upscale_start = old_source_str.find("class UpscaleBlock")
if upscale_start == -1:
    print("ERROR: Could not find 'class UpscaleBlock' in cell source")
    exit(1)

# Keep everything before UpscaleBlock
prefix = old_source_str[:upscale_start]

# New code: PixelShuffleUpscale + UpscaleBlock + LocallyConnected2D + ResidualLCB
new_classes = '''class PixelShuffleUpscale(nn.Module):
    """Conv → PixelShuffle → BN → ReLU up-sampling block."""
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch * (scale_factor ** 2),
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(scale_factor),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpscaleBlock(nn.Module):
    """
    6× upscale via sub-pixel convolution (PixelShuffle) + refinement conv.
    Input  : (B, C, 24, 32)   — low-res encoder output
    Output : (B, C, 144, 192) — high-res prediction
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up1 = PixelShuffleUpscale(in_ch, in_ch // 2, scale_factor=2)
        self.up2 = PixelShuffleUpscale(in_ch // 2, in_ch // 4, scale_factor=3)
        self.refine = nn.Conv2d(in_ch // 4, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up1(x)    # (B, C//2, 48,  64)
        x = self.up2(x)    # (B, C//4, 144, 192)
        return self.refine(x)


class LocallyConnected2D(nn.Module):
    """Position-specific 1x1 mixing. Use sparingly because it removes weight sharing."""
    def __init__(self, in_ch, out_ch, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.weight = nn.Parameter(torch.randn(height, width, out_ch, in_ch) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1, out_ch, height, width))

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == self.height, f"Height mismatch: got {h}, expected {self.height}"
        assert w == self.width, f"Width mismatch: got {w}, expected {self.width}"

        x = x.permute(0, 2, 3, 1).contiguous()              # (B, H, W, Cin)
        out = (x.unsqueeze(3) * self.weight.unsqueeze(0))   # (B, H, W, Cout, Cin)
        out = out.sum(dim=-1)                               # (B, H, W, Cout)
        out = out.permute(0, 3, 1, 2).contiguous()          # (B, Cout, H, W)
        return out + self.bias


class ResidualLCB(nn.Module):
    """
    Safer localized block:
      1x1 reduce -> local mixing -> 1x1 expand -> residual add

    This keeps the location-specific idea, but only at low spatial resolution
    and with fewer channels so it does not dominate the features.
    """
    def __init__(self, channels, height, width, reduction=4, min_hidden=32):
        super().__init__()
        hidden = min(channels, max(channels // reduction, min_hidden))

        self.pre = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.local = LocallyConnected2D(hidden, hidden, height, width)
        self.post = nn.Sequential(
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.pre(x)
        x = self.local(x)
        x = self.post(x)
        return self.out_act(x + residual)'''

new_source_str = prefix + new_classes
nb["cells"][upscale_cell_idx]["source"] = new_source_str.split("\n")
# Convert to list of lines with \n appended (except the last)
lines = new_source_str.split("\n")
nb["cells"][upscale_cell_idx]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
# Clear outputs and reset execution count
nb["cells"][upscale_cell_idx]["outputs"] = []
nb["cells"][upscale_cell_idx]["execution_count"] = None

print("  ✅ Replaced UpscaleBlock with PixelShuffle + added LCB classes")


# ============================================================
# 2. Find the ResNet18SR cell and update it
# ============================================================
resnet_cell_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "class ResNet18SR" in source:
            resnet_cell_idx = i
            break

if resnet_cell_idx is None:
    print("ERROR: Could not find ResNet18SR cell")
    exit(1)

print(f"Found ResNet18SR cell at index {resnet_cell_idx}")

old_resnet_source = nb["cells"][resnet_cell_idx]["source"]
old_resnet_str = "".join(old_resnet_source)

# 2a. Add output_lcb after sr_head in __init__
old_sr_head = "        # 6× Super-resolution head\n        self.sr_head = UpscaleBlock(64, out_ch)\n"
new_sr_head = (
    "        # 6× Super-resolution head\n"
    "        self.sr_head = UpscaleBlock(64, out_ch)\n"
    "        \n"
    "        # Locally Connected Block for position-specific spatial mixing\n"
    "        self.output_lcb = ResidualLCB(out_ch, height=144, width=192, reduction=4)\n"
)

if old_sr_head not in old_resnet_str:
    # Try with different unicode character
    old_sr_head = "        # 6\u00d7 Super-resolution head\n        self.sr_head = UpscaleBlock(64, out_ch)\n"
    new_sr_head = (
        "        # 6\u00d7 Super-resolution head\n"
        "        self.sr_head = UpscaleBlock(64, out_ch)\n"
        "        \n"
        "        # Locally Connected Block for position-specific spatial mixing\n"
        "        self.output_lcb = ResidualLCB(out_ch, height=144, width=192, reduction=4)\n"
    )

if old_sr_head in old_resnet_str:
    old_resnet_str = old_resnet_str.replace(old_sr_head, new_sr_head)
    print("  ✅ Added output_lcb to ResNet18SR.__init__")
else:
    print("WARNING: Could not find sr_head definition to add output_lcb")
    # Let's try to find what's actually there
    for line in old_resnet_source:
        if "sr_head" in line:
            print(f"  Found sr_head line: {repr(line)}")

# 2b. Update forward method to use output_lcb
old_forward_return = "        # 6× Super-resolution\n        return self.sr_head(x)                # (B, 4, 144, 192)"
new_forward_return = (
    "        # 6× Super-resolution\n"
    "        x = self.sr_head(x)                   # (B, 4, 144, 192)\n"
    "        \n"
    "        # Position-specific spatial mixing\n"
    "        return self.output_lcb(x)             # (B, 4, 144, 192)"
)

if old_forward_return not in old_resnet_str:
    # Try with unicode ×
    old_forward_return = "        # 6\u00d7 Super-resolution\n        return self.sr_head(x)                # (B, 4, 144, 192)"
    new_forward_return = (
        "        # 6\u00d7 Super-resolution\n"
        "        x = self.sr_head(x)                   # (B, 4, 144, 192)\n"
        "        \n"
        "        # Position-specific spatial mixing\n"
        "        return self.output_lcb(x)             # (B, 4, 144, 192)"
    )

if old_forward_return in old_resnet_str:
    old_resnet_str = old_resnet_str.replace(old_forward_return, new_forward_return)
    print("  ✅ Updated ResNet18SR.forward to use output_lcb")
else:
    print("WARNING: Could not find forward return to update")
    # Debug: print relevant lines
    for line in old_resnet_source:
        if "sr_head" in line or "Super-resolution" in line:
            print(f"  Found line: {repr(line)}")

# Update the model description
old_desc = "    Uses learnable Conv+Upsample blocks instead of F.interpolate.\n"
new_desc = "    Uses PixelShuffle upscaling + ResidualLCB for position-specific mixing.\n"
old_resnet_str = old_resnet_str.replace(old_desc, new_desc)

old_title = "# WITH UPSAMPLE BLOCKS + FINE-TUNING\n"
new_title = "# WITH PIXELSHUFFLE + LCB + FINE-TUNING\n"
old_resnet_str = old_resnet_str.replace(old_title, new_title)

# Convert back to list of lines
lines = old_resnet_str.split("\n")
nb["cells"][resnet_cell_idx]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
nb["cells"][resnet_cell_idx]["outputs"] = []
nb["cells"][resnet_cell_idx]["execution_count"] = None

# ============================================================
# 2c. Update saved scenario metadata so this variant is tracked separately
# ============================================================
config_cell_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "'scenario'" in source and "scenario2-resnet18-mae-perfect-prognosis" in source:
            config_cell_idx = i
            break

if config_cell_idx is None:
    print("WARNING: Could not find config cell to rename scenario metadata")
else:
    config_str = "".join(nb["cells"][config_cell_idx]["source"])
    config_str = config_str.replace(
        "scenario2-resnet18-mae-perfect-prognosis",
        "scenario2-resnet18-mae-pixelswitch-lcb-perfect-prognosis",
    )
    lines = config_str.split("\n")
    nb["cells"][config_cell_idx]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    nb["cells"][config_cell_idx]["outputs"] = []
    nb["cells"][config_cell_idx]["execution_count"] = None
    print("  ✅ Updated scenario metadata to the pixelswitch + LCB variant")

# ============================================================
# 3. Save the modified notebook
# ============================================================
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ Successfully saved modified notebook to {notebook_path}")
print("   Changes made:")
print("   1. Replaced bilinear UpscaleBlock with PixelShuffle-based UpscaleBlock")
print("   2. Added LocallyConnected2D class")
print("   3. Added ResidualLCB class")
print("   4. Integrated ResidualLCB into ResNet18SR model")
