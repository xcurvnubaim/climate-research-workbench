import os

path = r'c:\Users\zharif\Documents\Dataset TA\climate-research-workbench\notebooks\evaluation\refactor_all_notebooks.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    'sc2_eval_unet_mae_preupsample.ipynb': ('SRUNet', 'UNet'),
    'sc2_eval_covnext_mae.ipynb': ('SRUNet', 'ConvNeXtSR'),
    'sc2_eval_convnext_mae_preupsample.ipynb': ('SRUNet', 'ConvNeXtPreUpsample'),
    'sc2_eval_resnet18_mae.ipynb': ('SRUNet', 'ResNet18SR'),
    'sc2_eval_resnet18_mae_preupsample.ipynb': ('SRUNet', 'ResNet18PreUpsample'),
    'sc1_eval_convnext_mae.ipynb': ('SRUNet', 'ConvNeXtSR'),
    'sc1_eval_convnext_mae_pixel.ipynb': ('SRUNet', 'ConvNeXtSR'),
    'sc1_eval_convnext_mae_preupsample.ipynb': ('SRUNet', 'ConvNeXtPreUpsample'),
    'sc1_eval_resnet18.ipynb': ('SRUNet', 'ResNet18SR'),
    'sc1_eval_resnet18_preupsample.ipynb': ('SRUNet', 'ResNet18PreUpsample'),
    'sc1_eval_unet_preupsample.ipynb': ('SRUNet', 'UNet'),
}

lines = content.split('\n')
for i, line in enumerate(lines):
    if line.strip().startswith('"sc'):
        notebook_key = line.strip().split(':')[0].strip('"')
        
        # Look ahead for model_load
        for j in range(i+1, min(i+40, len(lines))):
            if lines[j].strip().startswith('"model_load"'):
                # it's on the next line or the string
                for k in range(j+1, min(j+10, len(lines))):
                    if 'model = ' in lines[k]:
                        if notebook_key in replacements:
                            old, new = replacements[notebook_key]
                            lines[k] = lines[k].replace(f'model = {old}', f'model = {new}')
                        break
                break

with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Successfully replaced class names in refactor_all_notebooks.py")
