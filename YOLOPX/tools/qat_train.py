import torch
from pytorch_quantization import quant_modules
# IMPORTANT: This must be called BEFORE importing the model
quant_modules.initialize()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.models import get_net
from lib.config import cfg
import lib.dataset as dataset
from lib.utils import DataLoaderX
from lib.core.loss import get_loss
import torchvision.transforms as transforms

def main():
    # 1. Setup Environment
    device = torch.device('cuda:0')
    cfg.defrost()
    cfg.TRAIN.BATCH_SIZE_PER_GPU = 4 # Kept low for 8GB VRAM
    cfg.DATASET.TRAIN_SET = 'val' # Hack: Train on the val set since we haven't converted the 70GB train set
    cfg.freeze()

    # 2. Initialize QAT Model
    print("Building QAT Model...")
    model = get_net(cfg)
    checkpoint = torch.load('YOLOPX/weights/epoch-195.pth', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)

    # 3. Prepare Data
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    dataloader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )

    # 4. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # Very low LR for fine-tuning
    criterion = get_loss(cfg, device, model)

    # 5. QAT Fine-Tuning Loop
    model.train()
    print("Starting QAT Fine-Tuning (1 Epoch)...")
    for epoch in range(1):
        for i, (imgs, targets, paths, shapes) in enumerate(dataloader):
            imgs = imgs.to(device).float()
            targets = [t.to(device) for t in targets]
            
            # Forward Pass (with Fake Quantization)
            outputs = model(imgs)
            loss, _ = criterion(outputs, targets, shapes, model, imgs)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/10] Batch [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        # Save QAT Checkpoint
        torch.save({'state_dict': model.state_dict()}, f'YOLOPX/weights/yolopx_qat_epoch_{epoch}.pth')
    
    print("QAT Fine-Tuning Complete!")

if __name__ == '__main__':
    main()