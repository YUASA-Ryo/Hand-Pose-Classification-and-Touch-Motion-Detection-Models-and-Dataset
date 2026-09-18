import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


#Datapath
DATA_PATH = os.path.join("data", "modelData2.npz")

#Hyperparameters
SEQUENCE_LENGTH = 30
EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 128
LEARNING_RATE = 8.7665324e-4
WEIGHT_DECAY = 4.4288308e-5
DENSE_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.2341727
THRESHOLD = 0.5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Seed
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


# Z Correction

FIXED_Z_THRESHOLD = 0.08
Z_LOW_GAIN = 1.0
Z_HIGH_GAIN = 5.0
Z_SMOOTH_ALPHA = 4.0

def z_smooth_gain_by_delta(delta_z, threshold):
    abs_delta = np.abs(delta_z)
    sigmoid = 1.0 / (1.0 + np.exp(-Z_SMOOTH_ALPHA * (abs_delta - threshold)))
    gain = Z_LOW_GAIN + (Z_HIGH_GAIN - Z_LOW_GAIN) * sigmoid
    return delta_z * gain


def apply_z_correction(z_values, threshold):
    base_z = z_values[0].copy()
    delta_z = z_values - base_z[None, :]
    delta_z = z_smooth_gain_by_delta(delta_z, threshold)
    return base_z[None, :] + delta_z

# Preprocessing

def preprocess_sample(sample):
    sample = np.asarray(sample, dtype=np.float32)

    coord_frames = []
    quat_frames = []

    for frame in sample:
        coords = frame[:81].reshape(27, 3)
        quat = frame[81:85]

        ref_coord = coords[-1]
        relative_coords = coords - ref_coord
        relative_coords = relative_coords[:-1]

        coord_frames.append(relative_coords)
        quat_frames.append(quat.copy())

    processed_frames = np.asarray(coord_frames, dtype=np.float32)
    quat_frames = np.asarray(quat_frames, dtype=np.float32)

    processed_frames[:, :, 2] = apply_z_correction(processed_frames[:, :, 2], FIXED_Z_THRESHOLD)

    processed_frames = processed_frames.reshape(processed_frames.shape[0], -1)

    processed_frames = np.concatenate([processed_frames, quat_frames], axis=1)

    velocity = np.zeros((processed_frames.shape[0], 1), dtype=np.float32)

    for t in range(1, processed_frames.shape[0]):
        velocity[t, 0] = processed_frames[t, 5] - processed_frames[t - 1, 5]

    processed_frames = np.concatenate([processed_frames, velocity], axis=1)

    processed_frames = processed_frames[-SEQUENCE_LENGTH:]

    return processed_frames

#Data
data = np.load(DATA_PATH, allow_pickle=True)

cursorFirst = data["cursorFirst"].tolist() #False(1)
cursorSecond = data["cursorSecond"].tolist() #True(0)
cursorManyFirst = data["cursorManyFirst"].tolist() #False(1)
cursorManySecond = data["cursorManySecond"].tolist() #True(0)

X_all_processed = []
y_all_processed = []
subject_all_processed = []

dataset_sources = [(cursorFirst, 1), (cursorSecond, 0), (cursorManyFirst, 1), (cursorManySecond, 0)]

subject_count = len(cursorFirst)

for subject_idx in range(subject_count):
    for data_list, label in dataset_sources:
        subject_samples = data_list[subject_idx]

        for sample in subject_samples:
            processed = preprocess_sample(sample)

            X_all_processed.append(processed)
            y_all_processed.append(label)
            subject_all_processed.append(subject_idx)


X_all = np.asarray(X_all_processed, dtype=np.float32)
y_all = np.asarray(y_all_processed, dtype=np.float32)
subject_all = np.asarray(subject_all_processed, dtype=np.int64)

print("X_all shape:", X_all.shape)
print("y_all shape:", y_all.shape)
print("subject_all shape:", subject_all.shape)
print("1 sample shape:", X_all[0].shape)
print("1 frame dimension:", X_all[0][0].shape[0])

#Split Data
def split_subject_data(subject_indices):
    rng = np.random.RandomState(SEED)

    shuffled = subject_indices.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)

    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)

    train_indices = shuffled[:train_size]
    val_indices = shuffled[train_size:train_size + val_size]
    test_indices = shuffled[train_size + val_size:]

    return train_indices, val_indices, test_indices


#Make Data
train_indices_all = []
val_indices_all = []
test_indices_all = []

subject_ids = np.unique(subject_all)

for subject in subject_ids:
    subject_indices = np.where(subject_all == subject)[0]

    train_indices, val_indices, test_indices = split_subject_data(subject_indices)

    train_indices_all.extend(train_indices)
    val_indices_all.extend(val_indices)
    test_indices_all.extend(test_indices)


train_indices_all = np.asarray(train_indices_all)
val_indices_all = np.asarray(val_indices_all)
test_indices_all = np.asarray(test_indices_all)


X_train = X_all[train_indices_all]
y_train = y_all[train_indices_all]

X_val = X_all[val_indices_all]
y_val = y_all[val_indices_all]

X_test = X_all[test_indices_all]
y_test = y_all[test_indices_all]


print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("Train labels:", y_train.shape)
print("Validation labels:", y_val.shape)
print("Test labels:", y_test.shape)


#Model

class Task2LSTM(nn.Module):

    def __init__(self, input_dim=83, hidden_dim=256, dense_dim=128, num_layers=3, dropout=0.23417273785978382):
        super().__init__()

        self.dense = nn.Linear(input_dim, dense_dim)

        self.lstm = nn.LSTM(
            input_size=dense_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.dense(x)

        output, _ = self.lstm(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32),torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)),
    batch_size=BATCH_SIZE,
    shuffle=False
)



model = Task2LSTM(input_dim=X_train.shape[2], hidden_dim=HIDDEN_DIM, dense_dim=DENSE_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def evaluate_loss(loader):
    model.eval()

    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

    return total_loss / total_n

best_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())
patience = 0

for epoch in range(EPOCHS):

    model.train()

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()

        optimizer.step()

    val_loss = evaluate_loss(val_loader)

    print(f"Epoch {epoch + 1:3d} / {EPOCHS}: val_loss = {val_loss:.6f}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        patience = 0
    else:
        patience += 1

        if patience >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

model.load_state_dict(best_state)
model.eval()

preds_all = []
true_all = []
prob_all = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)

        probs = torch.sigmoid(model(xb)).cpu().numpy()

        preds = (probs >= THRESHOLD).astype(int)

        preds_all.append(preds)
        true_all.append(yb.numpy().astype(int))
        prob_all.append(probs)


y_pred = np.vstack(preds_all)
y_true = np.vstack(true_all)
y_score = np.vstack(prob_all)


accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

fpr, tpr, _ = roc_curve(
    y_true.ravel(),
    y_score.ravel()
)



print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")