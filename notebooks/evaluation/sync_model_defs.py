import json
import os
import glob

EVAL_DIR = r"c:\Users\zharif\Documents\Dataset TA\climate-research-workbench\notebooks\evaluation"
TRAIN_DIR = r"c:\Users\zharif\Documents\Dataset TA\climate-research-workbench\notebooks\train\fix\runs"

# Map eval notebook to training notebook (run1)
MAPPING = {
    # SC1
    "sc1_eval_convnext_mae.ipynb": "sc1_convnext_mae_run1_executed.ipynb",
    "sc1_eval_convnext_mae_pixel.ipynb": "sc1_convnext_mae_pixel_run1_executed.ipynb",
    "sc1_eval_convnext_mae_preupsample.ipynb": "sc1_convnext_mae_preupsample_run1_executed.ipynb",
    "sc1_eval_gan.ipynb": "sc1_gan_run1_executed.ipynb",
    "sc1_eval_gan_preupsample.ipynb": "sc1_gan_preupsample_run1_executed.ipynb",
    "sc1_eval_resnet18.ipynb": "sc1_resnet18_run1_executed.ipynb",
    "sc1_eval_resnet18_preupsample.ipynb": "sc1_resnet18_preupsample_run1_executed.ipynb",
    "sc1_eval_unet.ipynb": "sc1_unet_run1_executed.ipynb",
    "sc1_eval_unet_preupsample.ipynb": "sc1_unet_preupsample_run1_executed.ipynb",
    
    # SC2
    "sc2_eval_gan.ipynb": "sc2_gan_run1_executed.ipynb",
    "sc2_eval_gan_preupsample.ipynb": "sc2_gan_preupsample_run1_executed.ipynb",
    "sc2_eval_unet_mae.ipynb": "sc2_unet_mae_run1_executed.ipynb",
    "sc2_eval_unet_mae_preupsample.ipynb": "sc2_unet_mae_preupsample_run1_executed.ipynb",
    "sc2_eval_covnext_mae.ipynb": "sc2_covnext_mae_run1_executed.ipynb",
    "sc2_eval_convnext_mae_preupsample.ipynb": "sc2_convnext_mae_preupsample_run1_executed.ipynb",
    "sc2_eval_resnet18_mae.ipynb": "sc2_resnet18_mae_run1_executed.ipynb",
    "sc2_eval_resnet18_mae_preupsample.ipynb": "sc2_resnet18_mae_preupsample_run1_executed.ipynb",
}

for eval_name, train_name in MAPPING.items():
    eval_path = os.path.join(EVAL_DIR, eval_name)
    train_path = os.path.join(TRAIN_DIR, train_name)
    
    if not os.path.exists(eval_path):
        print(f"Skipping {eval_name} (not found)")
        continue
    
    if not os.path.exists(train_path):
        print(f"WARNING: Train notebook {train_name} not found for {eval_name}")
        continue
        
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_nb = json.load(f)
        
    with open(train_path, "r", encoding="utf-8") as f:
        train_nb = json.load(f)
        
    # Find model definition cell in training notebook.
    # Usually it's cell 3, but let's be robust and look for a code cell containing 'class ' and 'nn.Module'.
    model_cell = None
    for cell in train_nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell.get("source", []))
            if "class " in src and "nn.Module" in src:
                # To distinguish it from datastructures, check for specific base model class names
                if any(x in src for x in ["SRUNet", "ConvNeXtSR", "ConvNeXtPreUpsample", "ResNet18SR", "ResNet18PreUpsample", "RRDBGenerator", "UNet"]):
                    model_cell = cell
                    break
    
    if not model_cell:
        # Fallback to cell 3 if search fails
        model_cell = train_nb["cells"][3]
        
    # Replace cell 3 in eval notebook with the model cell from training notebook
    # Ensure no outputs/execution_count are carried over
    model_cell["outputs"] = []
    model_cell["execution_count"] = None
    model_cell["id"] = "model_def_patched"
    
    eval_nb["cells"][3] = model_cell
    
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_nb, f, indent=1, ensure_ascii=False)
        
    print(f"Patched {eval_name} with model def from {train_name}")

