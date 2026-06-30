"""
Train the SEMSPEM WSD bi-encoder with a hierarchical weighted InfoNCE loss.

Encodes (context, target span) and WordNet glosses with a shared transformer,
and contrasts the gold sense against easy / semi-hard / hard negative senses
sampled from the WordNet sense inventory.
"""

import argparse
import os
import sys

import nltk
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

nltk.download('wordnet')

# Make the repo root and the Dataset/ and Model/ package dirs importable
# regardless of the working directory, so `python Model/Train_Bi_WSD.py` works
# after a fresh `git clone` on any platform.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT,
           os.path.join(_REPO_ROOT, "Dataset"),
           os.path.join(_REPO_ROOT, "Model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import BiEncoderWSDataset, wsd_collate_fn
from Loss import W_EASY, W_HARD, W_SEMI, hierarchical_infonce_loss
from model import SharedEncoder
from utils import build_wordnet_index


def parse_args():
    p = argparse.ArgumentParser(description="Train the SEMSPEM WSD bi-encoder")
    p.add_argument("--csv_path", default="./data/semcor_training_samples.csv",
                    help="SemCor training samples CSV")
    p.add_argument("--best_model_path", default="./checkpoints/best_bi_encoder_wsd.pt",
                    help="Where to save the best checkpoint")
    return p.parse_args()


ARGS = parse_args()


# 0. Configuration

CSV_PATH        = ARGS.csv_path
BEST_MODEL_PATH = ARGS.best_model_path

PRETRAINED      = "bert-base-uncased"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE      = 8
EPOCHS          = 6
LR              = 1e-5
VAL_FRACTION    = 0.2
TAU             = 0.1

# Negative sampling ratio
N_EASY, N_SEMI, N_HARD = 4, 2, 1



# 1. Build WordNet sense/gloss structures

SUPERSENSE2SYNS, LEMMA2SYNS, SYNSET2GLOSS = build_wordnet_index()



# 2. Train / validation step

def run_epoch(model: SharedEncoder, loader, optimizer, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="Train" if is_train else "Val "):
            if not batch:
                continue

            batch_loss = torch.tensor(0.0, device=DEVICE)

            for item in batch:
                f_anchor = model.encode_context(
                    item["tokens"], tuple(item["span"])
                )
                f_pos  = model.encode_gloss(SYNSET2GLOSS[item["gold_sid"]])
                f_easy = [model.encode_gloss(SYNSET2GLOSS[s]) for s in item["easy_negs"]]
                f_semi = [model.encode_gloss(SYNSET2GLOSS[s]) for s in item["semi_negs"]]
                f_hard = [model.encode_gloss(SYNSET2GLOSS[s]) for s in item["hard_negs"]]

                loss = hierarchical_infonce_loss(
                    f_anchor, f_pos, f_easy, f_semi, f_hard,
                    tau=TAU, w_e=W_EASY, w_s=W_SEMI, w_h=W_HARD
                )
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / len(batch)

            if is_train:
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += batch_loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)



# 3. Main

def main():
    print("Loading dataset...")
    dataset = BiEncoderWSDataset(
        CSV_PATH, SYNSET2GLOSS, SUPERSENSE2SYNS, LEMMA2SYNS,
        N_EASY, N_SEMI, N_HARD,
    )
    n_val   = int(len(dataset) * VAL_FRACTION)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=wsd_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=wsd_collate_fn)

    print("Initializing model (Shared Encoder)...")
    model     = SharedEncoder(PRETRAINED, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    sample_sid = next(iter(SYNSET2GLOSS))
    print("\nConfiguration summary:")
    print("  Gloss input format   'target word: definition'")
    print(f"  Input example        '{SYNSET2GLOSS[sample_sid]}'")
    print("  Weighting scheme     n_i x exp(1/(|L|-l+1))")
    print(f"  Negative ratio       Easy:Semi:Hard = {N_EASY}:{N_SEMI}:{N_HARD}")
    print(f"  Negative weights     w_e={W_EASY:.4f}, w_s={W_SEMI:.4f}, w_h={W_HARD:.4f}")
    print(f"  Effective contrib.   Easy={N_EASY*W_EASY:.2f}, "
          f"Semi={N_SEMI*W_SEMI:.2f}, Hard={N_HARD*W_HARD:.2f}")
    print(f"  Temperature          tau={TAU}\n")

    os.makedirs(os.path.dirname(BEST_MODEL_PATH) or ".", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, is_train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, is_train=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  Best model saved (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS + 1), val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Hierarchical InfoNCE (target word + gloss, exp weight)')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
