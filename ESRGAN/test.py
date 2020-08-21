import os.path as osp
import os
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

model_path = 'models/RRDB_PSNR_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

def walk(dir):
    for bpath in os.listdir(dir):
        path = os.path.join(dir, bpath)
        print(f"Glob root '{dir}': path '{path}'")
        if osp.isdir(path):
            yield from walk(path)
        else:
            yield path

idx = 0
for path in walk(test_img_folder):
    idx += 1
    parent = osp.dirname(path)
    basename = osp.basename(path).rsplit(".")[0]
    subdir = parent.split("/", 1)[-1]
    output_dir = os.path.join("results", subdir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{basename}_rlt.png")
    print(f"[{idx}] Path: {path}; Output: {output_path}")
    
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(output_path, output)
