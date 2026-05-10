import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import train_one_epoch, FocalLoss, make_weighted_sampler
from main_model import HiFuse_Small as create_model
from my_dataset import MyDataSet
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import os

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    num_classes = args.num_classes
    img_size = 224

    # ===== Data Augmentation =====
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    }

    train_dataset = MyDataSet(args.train_data_path, transform=data_transform["train"])
    val_dataset = MyDataSet(args.val_data_path, transform=data_transform["val"])

    # Weighted sampler
    sampler = None
    shuffle_train = True
    if args.use_weighted_sampler:
        sampler = make_weighted_sampler(train_dataset.label_list)
        shuffle_train = False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle_train, sampler=sampler,
                              pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # ===== Model =====
    model = create_model(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / 100))
    reduce_lr = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # Loss function
    loss_fn = FocalLoss(gamma=2.0) if args.use_focal else None

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n🚀 Epoch [{epoch}/{args.epochs}]")
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, epoch, warmup_scheduler, use_amp=True, loss_fn=loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, device, epoch)
        reduce_lr.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"./model_weight/best_model.pth")
            print(f"✅ Saved new best model (acc={best_acc:.3f})")

def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    accu_loss, accu_num, sample_num = 0.0, 0.0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            loss = loss_fn(pred, labels)
            accu_loss += loss.item()
            accu_num += torch.eq(torch.argmax(pred, 1), labels).sum().item()
            sample_num += images.size(0)

    val_loss = accu_loss / len(data_loader)
    val_acc = accu_num / sample_num
    print(f"[valid epoch {epoch}] loss: {val_loss:.3f}, acc: {val_acc:.3f}")
    return val_loss, val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-path', default='./data/train')
    parser.add_argument('--val-data-path', default='./data/val')
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--use-focal', action='store_true', help='Use Focal Loss')
    parser.add_argument('--use-weighted-sampler', action='store_true', help='Use WeightedRandomSampler')
    args = parser.parse_args()
    main(args)
