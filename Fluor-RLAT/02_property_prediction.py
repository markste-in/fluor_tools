#######################################################################
# Predict absorption wavelength
import pandas as pd
import numpy as np
import os
import random
import copy
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.data import MoleculeCSVDataset
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.neighbors import KernelDensity


if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

# Set global random seed
seed = 42
alpha = 0.1
epochs = 3
patience = 20
n_tasks = 1
graph_feat_size = 256 # Fixed
batch_size = 32       # Fixed
learning_rate = 1e-3  # Fixed


dropout = 0.3
num_layers = 2
num_timesteps = 2 


random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Use AttentiveFP featurizer
atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')
print("n_feats", n_feats, "e_feats", e_feats)

node_feat_size = n_feats
edge_feat_size = e_feats

def compute_lds_weights(targets, h=alpha, sigma=5, sqrt=False, amplify=False):
    targets = np.array(targets).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(targets)
    log_densities = kde.score_samples(targets)
    densities = np.exp(log_densities)
    weights = 1. / (densities ** h)
    if sqrt:
        weights = np.sqrt(weights)
    if amplify:
        median_val = np.median(targets)
        weights *= np.where(np.abs(targets - median_val) > 1.5, 2.0, 1.0)
    return torch.tensor(weights / np.mean(weights), dtype=torch.float32)


def load_data_with_fp(data, fp_data, name, load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='smiles',
                                 cache_file_path=str(name)+'_dataset_abs.bin',
                                 task_names=['abs'],
                                 load=load, init_mask=True, n_jobs=1)

    combined_data = []
    for i, data_tuple in enumerate(dataset):
        if len(data_tuple) == 3:
            smiles, graph, label = data_tuple
            mask = None
        else:
            smiles, graph, label, mask = data_tuple
        fp = torch.tensor(fp_data[i], dtype=torch.float32)
        combined_data.append((graph, fp, label, mask))
    return combined_data


# Load fingerprint data
def load_fingerprints(fp_file):
    df = pd.read_csv(fp_file)
    return torch.tensor(df.values, dtype=torch.float32)



# Load data
train_data = pd.read_csv('./data/train_abs.csv')
valid_data = pd.read_csv('./data/valid_abs.csv')

# Data standardization
scaler = StandardScaler()
train_data[['abs']] = scaler.fit_transform(train_data[['abs']])
valid_data[['abs']] = scaler.transform(valid_data[['abs']])

train_fp_solvent = load_fingerprints('./data/train_sol_abs.csv')
valid_fp_solvent = load_fingerprints('./data/valid_sol_abs.csv')
train_fp_smiles = load_fingerprints('./data/train_smiles_abs.csv')
valid_fp_smiles = load_fingerprints('./data/valid_smiles_abs.csv')

# === Extract extra features from train_data / valid_data (column indices 8:152) ===
train_fp_extra = torch.tensor(train_data.iloc[:, 8:152].values, dtype=torch.float32)
valid_fp_extra = torch.tensor(valid_data.iloc[:, 8:152].values, dtype=torch.float32)

# === Normalize numeric part (8 columns) ===
scaler_num = MinMaxScaler()

# Split: first 8 columns are numeric features, rest are supplementary fingerprints
train_num = train_fp_extra[:, :8].numpy()
valid_num = valid_fp_extra[:, :8].numpy()

train_rest = train_fp_extra[:, 8:]  # tensor latter part
valid_rest = valid_fp_extra[:, 8:]

# Fit and normalize first 8 columns
train_num_scaled = scaler_num.fit_transform(train_num)
valid_num_scaled = scaler_num.transform(valid_num)

# Convert back to tensor and concatenate
train_fp_extra = torch.cat([torch.tensor(train_num_scaled, dtype=torch.float32), train_rest], dim=1)
valid_fp_extra = torch.cat([torch.tensor(valid_num_scaled, dtype=torch.float32), valid_rest], dim=1)

# === Concatenate final features: solvent + smiles + extra ===
train_fp = torch.cat([train_fp_solvent, train_fp_smiles, train_fp_extra], dim=1)
valid_fp = torch.cat([valid_fp_solvent, valid_fp_smiles, valid_fp_extra], dim=1)


lds_weights = compute_lds_weights(train_data[['abs']].values.flatten())


class FingerprintAttentionCNN(nn.Module):
    def __init__(self, input_dim, conv_channels=64):
        super(FingerprintAttentionCNN, self).__init__()
        self.conv_feat = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.conv_attn = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, D]
        feat_map = self.conv_feat(x)         # [B, C, D]
        attn_map = self.conv_attn(x)         # [B, C, D]
        attn_weights = self.softmax(attn_map)
        attn_out = torch.sum(feat_map * attn_weights, dim=-1)  # [B, C]
        pooled = self.pool(feat_map).squeeze(-1)               # [B, C]
        return torch.cat([attn_out, pooled], dim=1)            # [B, 2C]



class GraphFingerprintsModel(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size,
                solvent_dim, smiles_extra_dim,  # Separate declarations for two fp input dimensions
                graph_feat_size=graph_feat_size, num_layers=num_layers, num_timesteps=num_timesteps,
                n_tasks=n_tasks, dropout=dropout):
        super(GraphFingerprintsModel, self).__init__()

        # Graph neural network part
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                edge_feat_size=edge_feat_size,
                                num_layers=num_layers,
                                graph_feat_size=graph_feat_size,
                                dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                        num_timesteps=num_timesteps,
                                        dropout=dropout)

        # Fingerprint part one: smiles + extra, using CNN-attention extraction
        self.fp_extractor = FingerprintAttentionCNN(smiles_extra_dim, conv_channels=graph_feat_size)

        # Fingerprint part two: solvent, using fully connected extraction
        self.solvent_extractor = nn.Sequential(
            nn.Linear(solvent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )


        total_input_dim = graph_feat_size + graph_feat_size + 2 * graph_feat_size
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )




    def forward(self, g, node_feats, edge_feats, fingerprints):
        if edge_feats is None or 'he' not in g.edata:
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, edge_feats.size(1)), device=g.device)

        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)  # [B, G]

        # === Separate fingerprints into three parts ===
        # Assume fingerprints.shape = [B, S + M + E] (solvent, smiles, extra)
        # You can split according to their respective dimensions
        B = fingerprints.size(0)
        solvent_feat = fingerprints[:, :train_fp_solvent.shape[1]]  # [B, S]
        smiles_extra_feat = fingerprints[:, train_fp_solvent.shape[1]:]  # [B, M+E]

        # Extract features separately
        solvent_out = self.solvent_extractor(solvent_feat)  # [B, G]
        smiles_extra_out = self.fp_extractor(smiles_extra_feat)  # [B, 2G]

        # Concatenate three part features
        combined_feats = torch.cat([graph_feats, solvent_out, smiles_extra_out], dim=1)  # [B, 3G]

        return self.predict(combined_feats)

# Custom dataset class
class MolecularDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Collate function for data loading
def collate_fn(batch):
    if len(batch[0]) == 5:
        graphs, fps, labels, masks, weights = zip(*batch)
        weights = torch.stack(weights)
    else:
        graphs, fps, labels, masks = zip(*batch)
        weights = None
    graphs = dgl.batch(graphs)
    fps = torch.stack(fps)
    labels = torch.stack(labels)
    masks = torch.stack(masks) if masks[0] is not None else None
    return graphs, fps, labels, masks, weights



# === Read target data ===
target_data = pd.read_csv('./input/input.csv')
target_fp_solvent = load_fingerprints('./input/target_sol_morgan.csv')
target_fp_smiles = load_fingerprints('./input/target_smiles_morgan.csv')

# === Numeric feature standardization ===
target_fp_extra = torch.tensor(target_data.iloc[:, 8:152].values, dtype=torch.float32)
target_num = target_fp_extra[:, :8].numpy()
target_rest = target_fp_extra[:, 8:]

# Load pretrained scaler_num
target_num_scaled = scaler_num.transform(target_num)
target_fp_extra = torch.cat([torch.tensor(target_num_scaled, dtype=torch.float32), target_rest], dim=1)

# === Concatenate final fingerprints ===
target_fp = torch.cat([target_fp_solvent, target_fp_smiles, target_fp_extra], dim=1)

# === Label standardization (only for interface consistency, not needed for prediction) ===
target_data[['abs']] = scaler.transform(target_data[['abs']])  # Note: only for dataset construction, does not affect prediction results

# === Construct dataset and dataloader ===
target_datasets = load_data_with_fp(target_data, target_fp, 'target', True)
target_dataset = MolecularDataset(target_datasets)
target_loader = DataLoader(target_dataset, batch_size=batch_size, collate_fn=collate_fn)




# Initialize model
solvent_dim = target_fp_solvent.shape[1]
smiles_extra_dim = target_fp_smiles.shape[1] + target_fp_extra.shape[1]

model = GraphFingerprintsModel(
    node_feat_size=n_feats,
    edge_feat_size=e_feats,
    solvent_dim=solvent_dim,
    smiles_extra_dim=smiles_extra_dim,
    graph_feat_size=graph_feat_size,
    num_layers=num_layers,
    num_timesteps=num_timesteps,
    n_tasks=n_tasks,
    dropout=dropout
).to(device)

# Load saved model parameters
model.load_state_dict(torch.load('Model_abs.pth', map_location=device))
model.eval()


# Prediction
def predict(model, dataloader):
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 5:
                graphs, fps, _, _, _ = batch  # Contains weights
            else:
                graphs, fps, _, _ = batch     # No weights
            graphs = graphs.to(device)
            fps = fps.to(device)
            node_feats = graphs.ndata['hv']
            edge_feats = graphs.edata['he']
            predictions = model(graphs, node_feats, edge_feats, fps)
            all_predictions.append(predictions.cpu().numpy())
    return np.vstack(all_predictions)


# Save prediction results to CSV file
def save_predictions(predictions, file_name):
    df = pd.DataFrame(predictions, columns=['abs'])
    df.to_csv(file_name, index=False)

# After prediction, reverse transform standardized prediction results
def reverse_standardization(predictions, scaler):
    return scaler.inverse_transform(predictions)


# === Model prediction ===
target_predictions = predict(model, target_loader)
target_scale_predictions = reverse_standardization(target_predictions, scaler)

# === Save prediction results ===
save_predictions(target_scale_predictions, './result/target_predictions_abs.csv')
print("🎯 Target predictions saved to target_predictions_abs.csv")






#######################################################################
# Predict emission wavelength

graph_feat_size = 256
alpha = 0
num_layers = 3
num_timesteps = 1
n_tasks = 1
dropout = 0.3
batch_size = 32
learning_rate = 1e-3
epochs = 3
patience = 20


random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



def load_data_with_fp(data, fp_data, name, load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='smiles',
                                 cache_file_path=str(name)+'_dataset_em.bin',
                                 task_names=['em'],
                                 load=load, init_mask=True, n_jobs=1)

    combined_data = []
    for i, data_tuple in enumerate(dataset):
        if len(data_tuple) == 3:
            smiles, graph, label = data_tuple
            mask = None
        else:
            smiles, graph, label, mask = data_tuple
        fp = torch.tensor(fp_data[i], dtype=torch.float32)
        combined_data.append((graph, fp, label, mask))
    return combined_data




# Load data
train_data = pd.read_csv('./data/train_em.csv')
valid_data = pd.read_csv('./data/valid_em.csv')

# Data standardization
scaler = StandardScaler()
train_data[['em']] = scaler.fit_transform(train_data[['em']])
valid_data[['em']] = scaler.transform(valid_data[['em']])

train_fp_solvent = load_fingerprints('./data/train_sol_em.csv')
valid_fp_solvent = load_fingerprints('./data/valid_sol_em.csv')
train_fp_smiles = load_fingerprints('./data/train_smiles_em.csv')
valid_fp_smiles = load_fingerprints('./data/valid_smiles_em.csv')

# === Extract extra features from train_data / valid_data (column indices 8:152) ===
train_fp_extra = torch.tensor(train_data.iloc[:, 8:152].values, dtype=torch.float32)
valid_fp_extra = torch.tensor(valid_data.iloc[:, 8:152].values, dtype=torch.float32)

# === Normalize numeric part (8 columns) ===
scaler_num = MinMaxScaler()

# Split: first 8 columns are numeric features, rest are supplementary fingerprints
train_num = train_fp_extra[:, :8].numpy()
valid_num = valid_fp_extra[:, :8].numpy()

train_rest = train_fp_extra[:, 8:]  # tensor latter part
valid_rest = valid_fp_extra[:, 8:]

# Fit and normalize first 8 columns
train_num_scaled = scaler_num.fit_transform(train_num)
valid_num_scaled = scaler_num.transform(valid_num)

# Convert back to tensor and concatenate
train_fp_extra = torch.cat([torch.tensor(train_num_scaled, dtype=torch.float32), train_rest], dim=1)
valid_fp_extra = torch.cat([torch.tensor(valid_num_scaled, dtype=torch.float32), valid_rest], dim=1)

# === Concatenate final features: solvent + smiles + extra ===
train_fp = torch.cat([train_fp_solvent, train_fp_smiles, train_fp_extra], dim=1)
valid_fp = torch.cat([valid_fp_solvent, valid_fp_smiles, valid_fp_extra], dim=1)
lds_weights = compute_lds_weights(train_data[['em']].values.flatten())



class GraphFingerprintsModel(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size,
                 solvent_dim, smiles_extra_dim,  # Separate declarations for two fp input dimensions
                 graph_feat_size=graph_feat_size, num_layers=num_layers, num_timesteps=num_timesteps,
                 n_tasks=n_tasks, dropout=dropout):
        super(GraphFingerprintsModel, self).__init__()

        # Graph neural network part
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)

        # Fingerprint part one: smiles + extra, using CNN-attention extraction
        self.fp_extractor = FingerprintAttentionCNN(smiles_extra_dim, conv_channels=graph_feat_size)

        # Fingerprint part two: solvent, using fully connected extraction
        self.solvent_extractor = nn.Sequential(
            nn.Linear(solvent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )

        total_input_dim = graph_feat_size + graph_feat_size + 2 * graph_feat_size
        # Final prediction after concatenation (3 * graph_feat_size)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, fingerprints):
        if edge_feats is None or 'he' not in g.edata:
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, edge_feats.size(1)), device=g.device)

        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)  # [B, G]

        # === Separate fingerprints into three parts ===
        # Assume fingerprints.shape = [B, S + M + E] (solvent, smiles, extra)
        # You can split according to their respective dimensions
        B = fingerprints.size(0)
        solvent_feat = fingerprints[:, :train_fp_solvent.shape[1]]  # [B, S]
        smiles_extra_feat = fingerprints[:, train_fp_solvent.shape[1]:]  # [B, M+E]

        # Extract features separately
        solvent_out = self.solvent_extractor(solvent_feat)  # [B, G]
        smiles_extra_out = self.fp_extractor(smiles_extra_feat)  # [B, 2G]

        # Concatenate three part features
        combined_feats = torch.cat([graph_feats, solvent_out, smiles_extra_out], dim=1)  # [B, 3G]

        return self.predict(combined_feats)



# === Read target data ===
target_data = pd.read_csv('./input/input.csv')
target_fp_solvent = load_fingerprints('./input/target_sol_morgan.csv')
target_fp_smiles = load_fingerprints('./input/target_smiles_morgan.csv')

# === Numeric feature standardization ===
target_fp_extra = torch.tensor(target_data.iloc[:, 8:152].values, dtype=torch.float32)
target_num = target_fp_extra[:, :8].numpy()
target_rest = target_fp_extra[:, 8:]

# Load pretrained scaler_num
target_num_scaled = scaler_num.transform(target_num)
target_fp_extra = torch.cat([torch.tensor(target_num_scaled, dtype=torch.float32), target_rest], dim=1)

# === Concatenate final fingerprints ===
target_fp = torch.cat([target_fp_solvent, target_fp_smiles, target_fp_extra], dim=1)

# === Label standardization (only for interface consistency, not needed for prediction) ===
target_data[['em']] = scaler.transform(target_data[['em']])  # Note: only for dataset construction, does not affect prediction results

# === Construct dataset and dataloader ===
target_datasets = load_data_with_fp(target_data, target_fp, 'target', True)
target_dataset = MolecularDataset(target_datasets)
target_loader = DataLoader(target_dataset, batch_size=batch_size, collate_fn=collate_fn)




# Initialize model
solvent_dim = target_fp_solvent.shape[1]
smiles_extra_dim = target_fp_smiles.shape[1] + target_fp_extra.shape[1]

model = GraphFingerprintsModel(
    node_feat_size=n_feats,
    edge_feat_size=e_feats,
    solvent_dim=solvent_dim,
    smiles_extra_dim=smiles_extra_dim,
    graph_feat_size=graph_feat_size,
    num_layers=num_layers,
    num_timesteps=num_timesteps,
    n_tasks=n_tasks,
    dropout=dropout
).to(device)

# Load saved model parameters
model.load_state_dict(torch.load('Model_em.pth', map_location=device))
model.eval()



# Save prediction results to CSV file
def save_predictions(predictions, file_name):
    df = pd.DataFrame(predictions, columns=['em'])
    df.to_csv(file_name, index=False)


# === 模型预测 ===
target_predictions = predict(model, target_loader)
target_scale_predictions = reverse_standardization(target_predictions, scaler)

# === 保存预测结果 ===
save_predictions(target_scale_predictions, './result/target_predictions_em.csv')
print("🎯 Target predictions saved to target_predictions_em.csv")






#######################################################################
# Predict quantum yield

graph_feat_size = 256
n_tasks = 1
dropout = 0.4
alpha = 0.2
batch_size = 32
learning_rate = 1e-3
epochs = 3
patience = 20
num_layers = 2
num_timesteps = 3

random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)




atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')


def load_data_with_fp(data, fp_data, name, load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='smiles',
                                 cache_file_path=str(name)+'_dataset_plqy.bin',
                                 task_names=['plqy'],
                                 load=load, init_mask=True, n_jobs=1)

    combined_data = []
    for i, data_tuple in enumerate(dataset):
        if len(data_tuple) == 3:
            smiles, graph, label = data_tuple
            mask = None
        else:
            smiles, graph, label, mask = data_tuple
        fp = torch.tensor(fp_data[i], dtype=torch.float32)
        combined_data.append((graph, fp, label, mask))
    return combined_data


def load_fingerprints(fp_file):
    df = pd.read_csv(fp_file)
    return torch.tensor(df.values, dtype=torch.float32)

train_data = pd.read_csv('./data/train_plqy.csv')
valid_data = pd.read_csv('./data/valid_plqy.csv')

scaler = StandardScaler()
train_data[['plqy']] = scaler.fit_transform(train_data[['plqy']])
valid_data[['plqy']] = scaler.transform(valid_data[['plqy']])

train_fp_solvent = load_fingerprints('./data/train_sol_plqy.csv')
valid_fp_solvent = load_fingerprints('./data/valid_sol_plqy.csv')
train_fp_smiles = load_fingerprints('./data/train_smiles_plqy.csv')
valid_fp_smiles = load_fingerprints('./data/valid_smiles_plqy.csv')

train_fp_extra = torch.tensor(train_data.iloc[:, 8:152].values, dtype=torch.float32)
valid_fp_extra = torch.tensor(valid_data.iloc[:, 8:152].values, dtype=torch.float32)
scaler_num = MinMaxScaler()
train_num = train_fp_extra[:, :8].numpy()
valid_num = valid_fp_extra[:, :8].numpy()
train_rest = train_fp_extra[:, 8:]
valid_rest = valid_fp_extra[:, 8:]
train_num_scaled = scaler_num.fit_transform(train_num)
valid_num_scaled = scaler_num.transform(valid_num)
train_fp_extra = torch.cat([torch.tensor(train_num_scaled, dtype=torch.float32), train_rest], dim=1)
valid_fp_extra = torch.cat([torch.tensor(valid_num_scaled, dtype=torch.float32), valid_rest], dim=1)
train_fp = torch.cat([train_fp_solvent, train_fp_smiles, train_fp_extra], dim=1)
valid_fp = torch.cat([valid_fp_solvent, valid_fp_smiles, valid_fp_extra], dim=1)


lds_weights = compute_lds_weights(train_data[['plqy']].values.flatten())

class GraphFingerprintsModel(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, fp_size, 
                 graph_feat_size=graph_feat_size, num_layers=num_layers, num_timesteps=num_timesteps, 
                 n_tasks=n_tasks, dropout=dropout):
        super(GraphFingerprintsModel, self).__init__()
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, fingerprints):
        if edge_feats is None or 'he' not in g.edata.keys():
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, e_feats)).to(g.device)
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        fp_feats = self.fp_fc(fingerprints)
        combined_feats = torch.cat([graph_feats, fp_feats], dim=1)
        return self.predict(combined_feats)

class MolecularDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    if len(batch[0]) == 5:
        graphs, fps, labels, masks, weights = zip(*batch)
        weights = torch.stack(weights)
    else:
        graphs, fps, labels, masks = zip(*batch)
        weights = None
    graphs = dgl.batch(graphs)
    fps = torch.stack(fps)
    labels = torch.stack(labels)
    masks = torch.stack(masks) if masks[0] is not None else None
    return graphs, fps, labels, masks, weights


# === Read target data ===
target_data = pd.read_csv('./input/input.csv')
target_fp_solvent = load_fingerprints('./input/target_sol_morgan.csv')
target_fp_smiles = load_fingerprints('./input/target_smiles_morgan.csv')

# === Numeric feature standardization ===
target_fp_extra = torch.tensor(target_data.iloc[:, 8:152].values, dtype=torch.float32)
target_num = target_fp_extra[:, :8].numpy()
target_rest = target_fp_extra[:, 8:]

# Load pretrained scaler_num
target_num_scaled = scaler_num.transform(target_num)
target_fp_extra = torch.cat([torch.tensor(target_num_scaled, dtype=torch.float32), target_rest], dim=1)

# === Concatenate final fingerprints ===
target_fp = torch.cat([target_fp_solvent, target_fp_smiles, target_fp_extra], dim=1)

# === Label standardization (only for interface consistency, not needed for prediction) ===
target_data[['plqy']] = scaler.transform(target_data[['plqy']])  # Note: only for dataset construction, does not affect prediction results

# === Construct dataset and dataloader ===
target_datasets = load_data_with_fp(target_data, target_fp, 'target', True)
target_dataset = MolecularDataset(target_datasets)
target_loader = DataLoader(target_dataset, batch_size=batch_size, collate_fn=collate_fn)


fp_size = target_fp.shape[1]
# Initialize model
model = GraphFingerprintsModel(node_feat_size=n_feats,
                               edge_feat_size=e_feats,
                               graph_feat_size=graph_feat_size,
                               num_layers=num_layers,
                               num_timesteps=num_timesteps,
                               fp_size=fp_size,
                               n_tasks=n_tasks,
                               dropout=dropout).to(device)

# Load saved model parameters
model.load_state_dict(torch.load('Model_plqy.pth', map_location=device))
model.eval()




# 将预测结果保存到 CSV 文件
def save_predictions(predictions, file_name):
    df = pd.DataFrame(predictions, columns=['plqy'])
    df.to_csv(file_name, index=False)


# === 模型预测 ===
target_predictions = predict(model, target_loader)
target_scale_predictions = reverse_standardization(target_predictions, scaler)

# === 保存预测结果 ===
save_predictions(target_scale_predictions, './result/target_predictions_plqy.csv')
print("🎯 Target predictions saved to target_predictions_plqy.csv")




#######################################################################
# Predict molar absorptivity

graph_feat_size = 256
n_tasks = 1
dropout = 0.3
alpha = 0.6
batch_size = 32
learning_rate = 1e-3
epochs = 3
patience = 20
num_layers = 3
num_timesteps = 1

random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)





atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')


def load_data_with_fp(data, fp_data, name, load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='smiles',
                                 cache_file_path=str(name)+'_dataset_k.bin',
                                 task_names=['k'],
                                 load=load, init_mask=True, n_jobs=1)

    combined_data = []
    for i, data_tuple in enumerate(dataset):
        if len(data_tuple) == 3:
            smiles, graph, label = data_tuple
            mask = None
        else:
            smiles, graph, label, mask = data_tuple
        fp = torch.tensor(fp_data[i], dtype=torch.float32)
        combined_data.append((graph, fp, label, mask))
    return combined_data


def load_fingerprints(fp_file):
    df = pd.read_csv(fp_file)
    return torch.tensor(df.values, dtype=torch.float32)

train_data = pd.read_csv('./data/train_k.csv')
valid_data = pd.read_csv('./data/valid_k.csv')

scaler = StandardScaler()
train_data[['k']] = scaler.fit_transform(train_data[['k']])
valid_data[['k']] = scaler.transform(valid_data[['k']])

train_fp_solvent = load_fingerprints('./data/train_sol_k.csv')
valid_fp_solvent = load_fingerprints('./data/valid_sol_k.csv')
train_fp_smiles = load_fingerprints('./data/train_smiles_k.csv')
valid_fp_smiles = load_fingerprints('./data/valid_smiles_k.csv')

train_fp_extra = torch.tensor(train_data.iloc[:, 8:152].values, dtype=torch.float32)
valid_fp_extra = torch.tensor(valid_data.iloc[:, 8:152].values, dtype=torch.float32)
scaler_num = MinMaxScaler()
train_num = train_fp_extra[:, :8].numpy()
valid_num = valid_fp_extra[:, :8].numpy()
train_rest = train_fp_extra[:, 8:]
valid_rest = valid_fp_extra[:, 8:]
train_num_scaled = scaler_num.fit_transform(train_num)
valid_num_scaled = scaler_num.transform(valid_num)
train_fp_extra = torch.cat([torch.tensor(train_num_scaled, dtype=torch.float32), train_rest], dim=1)
valid_fp_extra = torch.cat([torch.tensor(valid_num_scaled, dtype=torch.float32), valid_rest], dim=1)
train_fp = torch.cat([train_fp_solvent, train_fp_smiles, train_fp_extra], dim=1)
valid_fp = torch.cat([valid_fp_solvent, valid_fp_smiles, valid_fp_extra], dim=1)


lds_weights = compute_lds_weights(train_data[['k']].values.flatten())



class GraphFingerprintsModel(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, fp_size, 
                 graph_feat_size=graph_feat_size, num_layers=num_layers, num_timesteps=num_timesteps, 
                 n_tasks=n_tasks, dropout=dropout):
        super(GraphFingerprintsModel, self).__init__()
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, fingerprints):
        if edge_feats is None or 'he' not in g.edata.keys():
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, e_feats)).to(g.device)
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        fp_feats = self.fp_fc(fingerprints)
        combined_feats = torch.cat([graph_feats, fp_feats], dim=1)
        return self.predict(combined_feats)

class MolecularDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    if len(batch[0]) == 5:
        graphs, fps, labels, masks, weights = zip(*batch)
        weights = torch.stack(weights)
    else:
        graphs, fps, labels, masks = zip(*batch)
        weights = None
    graphs = dgl.batch(graphs)
    fps = torch.stack(fps)
    labels = torch.stack(labels)
    masks = torch.stack(masks) if masks[0] is not None else None
    return graphs, fps, labels, masks, weights



# === Read target data ===
target_data = pd.read_csv('./input/input.csv')
target_fp_solvent = load_fingerprints('./input/target_sol_morgan.csv')
target_fp_smiles = load_fingerprints('./input/target_smiles_morgan.csv')

# === Numeric feature standardization ===
target_fp_extra = torch.tensor(target_data.iloc[:, 8:152].values, dtype=torch.float32)
target_num = target_fp_extra[:, :8].numpy()
target_rest = target_fp_extra[:, 8:]

# Load pretrained scaler_num
target_num_scaled = scaler_num.transform(target_num)
target_fp_extra = torch.cat([torch.tensor(target_num_scaled, dtype=torch.float32), target_rest], dim=1)

# === Concatenate final fingerprints ===
target_fp = torch.cat([target_fp_solvent, target_fp_smiles, target_fp_extra], dim=1)

# === Label standardization (only for interface consistency, not needed for prediction) ===
target_data[['abs']] = scaler.transform(target_data[['abs']])  # Note: only for dataset construction, does not affect prediction results

# === Construct dataset and dataloader ===
target_datasets = load_data_with_fp(target_data, target_fp, 'target', True)
target_dataset = MolecularDataset(target_datasets)
target_loader = DataLoader(target_dataset, batch_size=batch_size, collate_fn=collate_fn)



fp_size = target_fp.shape[1]
# Initialize model
model = GraphFingerprintsModel(node_feat_size=n_feats,
                               edge_feat_size=e_feats,
                               graph_feat_size=graph_feat_size,
                               num_layers=num_layers,
                               num_timesteps=num_timesteps,
                               fp_size=fp_size,
                               n_tasks=n_tasks,
                               dropout=dropout).to(device)

# Load saved model parameters
model.load_state_dict(torch.load('Model_k.pth', map_location=device))
model.eval()




# 将预测结果保存到 CSV 文件
def save_predictions(predictions, file_name):
    df = pd.DataFrame(predictions, columns=['k'])
    df.to_csv(file_name, index=False)


# === 模型预测 ===
target_predictions = predict(model, target_loader)
target_scale_predictions = reverse_standardization(target_predictions, scaler)

# === 保存预测结果 ===
save_predictions(target_scale_predictions, './result/target_predictions_k.csv')
print("🎯 Target predictions saved to target_predictions_k.csv")