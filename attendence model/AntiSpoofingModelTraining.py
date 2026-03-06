# train_antispoof_dl.py
import os
import copy
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# -------------------------
# Config / Hyperparameters
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dataset", help="root folder with 'real' and 'spoof' subfolders")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--img_size", type=int, default=160)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--ensemble", action="store_true", help="If set, train ensemble (mobilenetv2 + resnet18). Otherwise trains mobilenetv2 only.")
parser.add_argument("--save_prefix", type=str, default="antispoof", help="prefix for saved model files")
parser.add_argument("--val_split", type=float, default=0.15)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Transforms (realistic)
# -------------------------
train_transforms = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # ±15 deg
    transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
    # GaussianBlur: kernel_size must be odd and >= 3. We vary sigma between 0.5 and 1.5
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.5,1.5))], p=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------
# Dataset & Split
# -------------------------
# Directory structure expected:
# dataset/
#   real/
#   spoof/
# Note: For best generalization split by subject (if filenames or folder structure encode subject).
dataset = datasets.ImageFolder(args.data_dir, transform=train_transforms)
class_to_idx = dataset.class_to_idx
print("Class to idx:", class_to_idx)
num_total = len(dataset)
val_count = int(num_total * args.val_split)
train_count = num_total - val_count
train_dataset, val_dataset = random_split(dataset, [train_count, val_count],
                                          generator=torch.Generator().manual_seed(42))
# Replace val transforms with val_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)


print(f"Total images: {num_total}, Train: {train_count}, Val: {val_count}")

# -------------------------
# Model builder
# -------------------------
def build_mobilenet(num_classes=2, pretrained=True):
    model = models.mobilenet_v2(pretrained=pretrained)
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def build_resnet18(num_classes=2, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# Freeze backbone if desired (transfer learning)
def freeze_backbone(model, freeze=True):
    if freeze:
        for name, param in model.named_parameters():
            # unfreeze classifier/fc layers
            if "classifier" in name or "fc" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    return model

# -------------------------
# Training loop utilities
# -------------------------
def train_one_epoch(models_dict, optimizers, criterions, loaders, epoch):
    """models_dict: dict name->model, optimizers: dict name->opt, criterions: dict name->loss
       loaders: (train_loader, val_loader)
    """
    train_loader, _ = loaders
    for model in models_dict.values():
        model.train()

    epoch_losses = {name: 0.0 for name in models_dict.keys()}
    correct = {name: 0 for name in models_dict.keys()}
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        total += labels.size(0)

        for name, model in models_dict.items():
            optimizers[name].zero_grad()
            outputs = model(images)
            loss = criterions[name](outputs, labels)
            loss.backward()
            optimizers[name].step()

            epoch_losses[name] += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct[name] += (preds == labels).sum().item()

    for name in models_dict.keys():
        epoch_losses[name] /= len(train_loader.dataset)
    return epoch_losses, {name: correct[name] / total for name in correct}

def evaluate_ensemble(models_dict, loader):
    for model in models_dict.values():
        model.eval()

    all_labels = []
    all_probs_ensemble = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            all_labels.append(labels.cpu().numpy())

            # Sum softmax probs from models (simple averaging ensemble)
            probs_sum = None
            for model in models_dict.values():
                logits = model(images)
                probs = torch.softmax(logits, dim=1)  # (B, 2)
                if probs_sum is None:
                    probs_sum = probs
                else:
                    probs_sum = probs_sum + probs
            probs_avg = probs_sum / len(models_dict)  # average
            all_probs_ensemble.append(probs_avg.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs_ensemble, axis=0)  # shape (N,2)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob[:,1])
    except Exception:
        auc = None
    report = classification_report(y_true, y_pred, target_names=list(class_to_idx.keys()), zero_division=0)
    return acc, auc, report

# -------------------------
# Build & prepare models
# -------------------------
models_to_train = {}
if args.ensemble:
    print("Building ensemble: mobilenetv2 + resnet18")
    model_a = build_mobilenet(num_classes=2, pretrained=True)
    model_b = build_resnet18(num_classes=2, pretrained=True)
    # Optionally freeze many layers for faster fine-tuning
    model_a = freeze_backbone(model_a, freeze=False)
    model_b = freeze_backbone(model_b, freeze=False)
    models_to_train["mobilenetv2"] = model_a.to(device)
    models_to_train["resnet18"] = model_b.to(device)
else:
    print("Building single model: mobilenetv2")
    model_a = build_mobilenet(num_classes=2, pretrained=True)
    model_a = freeze_backbone(model_a, freeze=False)
    models_to_train["mobilenetv2"] = model_a.to(device)

# Optimizers & criterions per model
optimizers = {}
criterions = {}
for name, model in models_to_train.items():
    # Only params with requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    optimizers[name] = optim.Adam(params, lr=args.lr)
    criterions[name] = nn.CrossEntropyLoss()

# -------------------------
# Training main loop
# -------------------------
best_val_acc = 0.0
best_models_state = {name: None for name in models_to_train.keys()}

for epoch in range(1, args.epochs + 1):
    train_losses, train_acc = train_one_epoch(models_to_train, optimizers, criterions, (train_loader, val_loader), epoch)
    print(f"Epoch {epoch} train losses: {train_losses}, train accs: {train_acc}")

    val_acc, val_auc, val_report = evaluate_ensemble(models_to_train, val_loader)
    print(f"Validation Acc: {val_acc:.4f}, AUC: {val_auc}")
    print("Classification Report:\n", val_report)

    # Save best
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        for name, model in models_to_train.items():
            best_models_state[name] = copy.deepcopy(model.state_dict())
        print(f"New best validation acc: {best_val_acc:.4f} - models saved to memory")

# -------------------------
# Save final models (as .pth and a combined .pkl)
# -------------------------
os.makedirs("trained_models", exist_ok=True)
for name, state in best_models_state.items():
    if state is None:
        print(f"Warning: model {name} has no saved state (maybe training failed)")
        continue
    pth_path = os.path.join("trained_models", f"{args.save_prefix}_{name}.pth")
    torch.save(state, pth_path)
    print(f"Saved {name} state_dict -> {pth_path}")

# Save entire ensemble to a single .pkl file (this pickles the state dicts and metadata)
ensemble_obj = {
    "models": {name: best_models_state[name] for name in best_models_state},
    "class_to_idx": class_to_idx,
    "img_size": args.img_size,
    "archs": list(best_models_state.keys())
}
pkl_path = os.path.join("trained_models", f"{args.save_prefix}_ensemble.pkl")
torch.save(ensemble_obj, pkl_path)
print(f"Saved ensemble metadata + state_dicts -> {pkl_path}")

# Optionally also save entire model objects (not recommended for long-term portability),
# but user asked for a pkl, so also save full model objects pickled (CPU-friendly)
full_models = {}
for name, model in models_to_train.items():
    # load best weights into a fresh model of same architecture to save full object
    if name == "mobilenetv2":
        m = build_mobilenet(num_classes=2, pretrained=False)
    elif name == "resnet18":
        m = build_resnet18(num_classes=2, pretrained=False)
    else:
        raise RuntimeError("Unknown model name")
    m.load_state_dict(best_models_state[name])
    m = m.to("cpu")
    full_models[name] = m

full_pkl_path = os.path.join("trained_models", f"{args.save_prefix}_fullmodels.pkl")
torch.save(full_models, full_pkl_path)
print(f"Saved pickled full model objects -> {full_pkl_path}")

print("Training complete. Best validation accuracy:", best_val_acc)
