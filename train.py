import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import HybridPneumoniaModel


def train_model():

    print("=" * 60)
    print("      STARTING HYBRID MODEL TRAINING PIPELINE")
    print("=" * 60)

    # =====================================================
    # 1. DATA TRANSFORMS
    # =====================================================

    data_transforms = {

        'train': transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(7),

            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]),

        'val': transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
    }

    # =====================================================
    # 2. DATASET PATH
    # =====================================================

    data_dir = 'dataset'

    if not os.path.exists(data_dir):
        print("[!] ERROR: dataset folder not found.")
        return

    if not os.path.exists(os.path.join(data_dir, 'train')):
        print("[!] ERROR: dataset/train not found.")
        return

    if not os.path.exists(os.path.join(data_dir, 'val')):
        print("[!] ERROR: dataset/val not found.")
        return

    # =====================================================
    # 3. LOAD DATASET
    # =====================================================

    try:

        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(data_dir, x),
                data_transforms[x]
            )
            for x in ['train', 'val']
        }

        dataloaders = {

            'train': DataLoader(
                image_datasets['train'],
                batch_size=16,
                shuffle=True,
                num_workers=0
            ),

            'val': DataLoader(
                image_datasets['val'],
                batch_size=16,
                shuffle=False,
                num_workers=0
            )
        }

        dataset_sizes = {
            x: len(image_datasets[x])
            for x in ['train', 'val']
        }

        print(f"[OK] Training samples   : {dataset_sizes['train']}")
        print(f"[OK] Validation samples : {dataset_sizes['val']}")

        print(
            f"[OK] Classes             : "
            f"{image_datasets['train'].classes}"
        )

    except Exception as e:

        print(f"[!] Dataset Loading Error: {e}")
        return

    # =====================================================
    # 4. DEVICE
    # =====================================================

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print(f"[OK] Training Device: {device}")

    # =====================================================
    # 5. MODEL
    # =====================================================

    model = HybridPneumoniaModel(
        num_classes=2
    ).to(device)

    print("[OK] Pretrained DenseNet-121 + Transformer loaded.")

    # =====================================================
    # 6. LOSS FUNCTION
    # =====================================================

    criterion = nn.CrossEntropyLoss()

    # =====================================================
    # 7. OPTIMIZER
    # =====================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # =====================================================
    # 8. LEARNING RATE SCHEDULER
    # =====================================================

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    # =====================================================
    # 9. TRAINING SETTINGS
    # =====================================================

    epochs = 20

    best_acc = 0.0

    # =====================================================
    # 10. TRAINING LOOP
    # =====================================================

    for epoch in range(epochs):

        print()
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 40)

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(
                    phase == 'train'
                ):

                    outputs = model(inputs)

                    loss = criterion(
                        outputs,
                        labels
                    )

                    _, preds = torch.max(
                        outputs,
                        1
                    )

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                running_loss += (
                    loss.item() * inputs.size(0)
                )

                running_corrects += torch.sum(
                    preds == labels.data
                )

            epoch_loss = (
                running_loss /
                dataset_sizes[phase]
            )

            epoch_acc = (
                running_corrects.double() /
                dataset_sizes[phase]
            )

            print(
                f"{phase.capitalize()} "
                f"Loss: {epoch_loss:.4f} "
                f"Acc: {epoch_acc:.4f}"
            )

            # =================================================
            # UPDATE SCHEDULER AFTER VALIDATION
            # =================================================

            if phase == 'val':

                scheduler.step(epoch_loss)

                current_lr = optimizer.param_groups[0]['lr']

                print(
                    f"Learning Rate: {current_lr:.7f}"
                )

                # =============================================
                # SAVE BEST MODEL
                # =============================================

                if epoch_acc > best_acc:

                    best_acc = epoch_acc

                    torch.save(
                        model.state_dict(),
                        'pneumonia_hybrid_model_new.pth'
                    )

                    print(
                        f"[OK] Best checkpoint saved!"
                    )

                    print(
                        f"[OK] Best Validation Accuracy: "
                        f"{best_acc:.4f}"
                    )

    # =====================================================
    # TRAINING COMPLETE
    # =====================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best Validation Accuracy: "
        f"{best_acc:.4f}"
    )

    print(
        "Model saved as: "
        "pneumonia_hybrid_model_new.pth"
    )

    print("=" * 60)


if __name__ == '__main__':
    train_model()
