import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Datapath
DATA_PATH = os.path.join("data", "modelData1.npz")

#Hyperparameters
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 5
LEARNING_RATE = 0.0095723
WEIGHT_DECAY = 2.6496815e-06
DROPOUT = 0.1776815
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Seed

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#Preprocess

def rotate_vector_inverse(v, q):
    q = q / (np.linalg.norm(q) + 1e-8)
    x, y, z, w = q
    q_vec = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(q_vec, v)
    rotated = v + w * t + np.cross(q_vec, t)
    return rotated


def preprocess_task1_frame(frame):
    coords = frame[:81].reshape(27, 3)
    ref_coord = coords[1]
    ref_quat = frame[81:85]
    processed = []
    for joint in range(2, 26):
        relative = coords[joint] - ref_coord
        rotated = rotate_vector_inverse(relative, ref_quat)
        processed.extend(rotated)

    return np.asarray(processed, dtype=np.float32)


def preprocess_dataset(data):
    processed_data = np.empty((len(data), 72), dtype=np.float32)

    for i, frame in enumerate(data):
        processed_data[i] = preprocess_task1_frame(frame)

    return processed_data

#Split Data

def split_subject_data(data, label, rng):
    indices = np.arange(len(data))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_x = data[train_indices]
    val_x = data[val_indices]
    test_x = data[test_indices]

    train_y = np.full(len(train_indices), label, dtype=np.float32)
    val_y = np.full(len(val_indices), label, dtype=np.float32)
    test_y = np.full(len(test_indices), label, dtype=np.float32)

    return train_x, train_y, val_x, val_y, test_x, test_y


def load_and_split_data():
    data = np.load(DATA_PATH, allow_pickle=True)

    #Key

    cursor_first = data["cursorFirst"] #True(0)
    cursor_second = data["cursorSecond"] #True(0)
    cursor_many_first = data["cursorManyFirst"] #False(1)
    cursor_many_second = data["cursorManySecond"] #False(1)

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    rng = np.random.default_rng(SEED)
    num_subjects = len(cursor_first)

    for subject_idx in range(num_subjects):
        subject_datasets = [(cursor_first[subject_idx], 0), (cursor_second[subject_idx], 0), (cursor_many_first[subject_idx], 1), (cursor_many_second[subject_idx], 1)]

        for subject_data, label in subject_datasets:
            subject_data = np.asarray(subject_data, dtype=np.float32)
            processed_data = preprocess_dataset(subject_data)
            tr_x, tr_y, va_x, va_y, te_x, te_y = split_subject_data(processed_data, label, rng)

            train_x.append(tr_x)
            train_y.append(tr_y)
            val_x.append(va_x)
            val_y.append(va_y)
            test_x.append(te_x)
            test_y.append(te_y)

    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    val_x = np.concatenate(val_x, axis=0)
    val_y = np.concatenate(val_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    train_indices = np.arange(len(train_x))
    val_indices = np.arange(len(val_x))

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_x = train_x[train_indices]
    train_y = train_y[train_indices]
    val_x = val_x[val_indices]
    val_y = val_y[val_indices]

    return train_x, train_y, val_x, val_y, test_x, test_y

#Model

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(72, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def make_loader(x, y, shuffle):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_model(train_x, train_y, val_x, val_y):
    train_loader = make_loader(train_x, train_y, True)
    val_loader = make_loader(val_x, val_y, False)

    model = MLP().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                output = model(x)
                loss = criterion(output, y)

                val_loss += loss.item() * len(x)
                val_count += len(x)

        val_loss /= val_count

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model


def evaluate(model, test_x, test_y):
    test_loader = make_loader(test_x, test_y, False)

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)

            output = model(x)
            probability = torch.sigmoid(output)
            prediction = (probability >= 0.5).int()

            all_predictions.extend(prediction.cpu().numpy().flatten())
            all_labels.extend(y.numpy().flatten().astype(int))

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    return accuracy, precision, recall, f1

# Main

def main():
    set_seed(SEED)

    train_x, train_y, val_x, val_y, test_x, test_y = load_and_split_data()

    print("Train:", train_x.shape)
    print("Validation:", val_x.shape)
    print("Test:", test_x.shape)
    print()

    model = train_model(train_x, train_y, val_x, val_y)
    accuracy, precision, recall, f1 = evaluate(model, test_x, test_y)

    print("Test Results")
    print()
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")


if __name__ == "__main__":
    main()