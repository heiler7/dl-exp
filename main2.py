import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import lr_scheduler
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer
from datetime import datetime
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=1, output_dim=1,batch_first=True):
        super(LSTM, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,num_layers=2,bidirectional=True)
        self.f_model = nn.Sequential(
            nn.Linear(360, 256), 
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, output_dim),

            )
 
    def forward(self, x):
        with torch.no_grad():
            x1 = torch.narrow(x, dim=-1, start=4, length=5)[:, :, ].squeeze(1)
            min_vals, _ = torch.min(x1, dim=1, keepdim=True)
            max_vals, _ = torch.max(x1, dim=1, keepdim=True)
            x1 = (x1 - min_vals) / (max_vals - min_vals +1)
            x1 = x1.unsqueeze(1)
            x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1] * x1.shape[2] * x1.shape[3]))
            
            yield1 = torch.narrow(x, dim=-1, start=5, length=1)[:, :, ].squeeze(1)
            yield1 = torch.reshape(yield1, (yield1.shape[0], yield1.shape[1] * yield1.shape[2]))
            
        apply = torch.narrow(x, dim=-1, start=2, length=1)[:, :, ].squeeze(1)
        redeem = torch.narrow(x, dim=-1, start=3, length=1)[:, :, ].squeeze(1)
        apply, _ = self.lstm(apply)
        redeem, _ = self.lstm(redeem)
        apply = torch.reshape(apply, (apply.shape[0], apply.shape[1] * apply.shape[2]))
        redeem = torch.reshape(redeem, (redeem.shape[0], redeem.shape[1] * redeem.shape[2]))
        yield2 = torch.narrow(x, dim=-1, start=5, length=1)[:, :, ].squeeze(1)
        yield2, _ = self.lstm(yield2)
        yield2 = torch.reshape(yield2, (yield2.shape[0], yield2.shape[1] * yield2.shape[2]))
        xin = torch.cat((apply,redeem,x1, yield1,yield2), dim=1)
        # out, _ = self.lstm(x)
        # out = torch.reshape(out, (out.shape[0], out.shape[1] * out.shape[2]))
        out = self.f_model(xin)
        out = torch.reshape(out, (out.shape[0], 10, 2))
        return out

def predict(path,predict_input,model):
    results = {}
    df = pd.read_csv(path)
    print(df.columns)
    products = df['product_pid'].unique()
    model.eval()
    with torch.no_grad():
        for i in products:
            data = predict_input[i]
            data = np.array(data).astype(np.float32)
            data = data[-20:,:]
            data = torch.tensor(data, dtype=torch.float32)
            data = data.reshape(1,data.shape[0],data.shape[1])
            # output = model(data.to(device))
            output = model(data.to(device))
            output = output.squeeze(0)
            results[i] = np.array(output.cpu())

    return results


def timechange(time):
    base = int(datetime.strptime(str(20210104), "%Y%m%d").timestamp())+58320000
    change = (int(datetime.strptime(str(time), "%Y%m%d").timestamp())-base)/86400
    return change

df = pd.read_csv(r'./data/product_info_simple_final_train_1.csv')
predict_path = './data/predict_input.csv'

groups = df.groupby('product_pid')
samples = []

predict_input = {}
predict_trans_input = {}
for _, group in groups:
    product_samples = group.values
    num_samples = len(product_samples)
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    product_name = product_samples[0][0]
        
    templst = []
    
    for j in range(num_samples):
        product_samples[j,1] = timechange(product_samples[j,1])
        product_samples[j,7] = timechange(product_samples[j,7])
        product_samples[j,8] = timechange(product_samples[j,8])
        product_samples[j,0] = product_samples[j,0].replace('product','')
        column_redeem_temp = product_samples[j,3]
        column_apply_temp = product_samples[j,2]
        if np.count_nonzero(column_redeem_temp) == 1 or np.count_nonzero(column_apply_temp) == 1:
            templst.append(product_samples[j])
    
    if num_samples < 30:
        continue
    
    predict_trans_input[product_name] = templst
    
    
    for i in range(num_samples - 29):
        sample = product_samples[i:i + 30, :]
        column_redeem_temp = sample[-10:, 3]
        column_apply_temp = sample[-10:, 2]
        
        are_elements_same = not (np.all(column_apply_temp != column_apply_temp[0]) or np.all(column_redeem_temp != column_redeem_temp[0]))
        if are_elements_same and np.count_nonzero(column_redeem_temp) == column_redeem_temp.size or np.count_nonzero(column_apply_temp) == column_apply_temp.size:
            samples.append(sample)
    
    predict_input[product_name] = samples[-1]
    
    # for i in range(num_samples):
    #     sample = product_samples[i, :]
        
    #     column_redeem_temp = sample[3]
    #     column_apply_temp = sample[2]

    #     if np.count_nonzero(column_redeem_temp) == 1 and np.count_nonzero(column_apply_temp) == 1:
    #         samples.append([sample])

final_array = np.array(samples)
# temp = np.concatenate((final_array[: , : , 5:7].astype(np.float32),final_array[: , : , 9:].astype(np.float32)),axis=2)
# train_data = np.concatenate((final_array[: , : , :2].astype(np.float32),temp),axis=2)
# train_data = final_array[: , : , :].astype(np.float32)
train_data = final_array[: , : , :].astype(np.float32)
train_label = final_array[: , : , 2:4].astype(np.float32)

# mean_values = np.mean(train_label, axis=1)
# mean_broadcasted = np.tile(mean_values[:, np.newaxis, :], (1, 10, 1))
# train_label = (train_label - mean_broadcasted)/(mean_broadcasted)

data = torch.tensor(train_data, dtype=torch.float32)
labels = torch.tensor(train_label, dtype=torch.float32)

batch_size = 500
learning_rate = 0.001
N = data.shape[0]
num_epochs = 49
k = 5
input_dim = 14
output_dim = 2
target_input_dim = 2

Lk = []
Lk_v = []
results_all = []
# 交叉验证


dataset_train = TensorDataset(data, labels)
train_loader = DataLoader(dataset_train, shuffle=False, batch_size=batch_size)

model = LSTM(input_dim=1, hidden_dim=2, output_dim=20)
model.to(device)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), weight_decay=1e-5)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)


L_train = []
L_val = []

min_validation_loss = 9999
for epoch in range(num_epochs):
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    train_running_loss = 0.0

    counter = 0
    model.train()
    for data_t, y in tqdm(train_loader):
        optimizer.zero_grad()
        counter += 1
        output = model(data_t.to(device))
        
        loss = criterion(output, y.to(device))
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    TL = train_running_loss / counter
    L_train.append(TL)
    model.eval()
    predict_res = []
    labels_true = []

    counter = 0
    
    print("Train loss: ", TL)
    
Lk.append(L_train)
Lk_v.append(L_val)

results1 = predict(predict_path,predict_input,model)
results_all.append(results1)

fig1, ax1 = plt.subplots()
for i, fold_train_loss in enumerate(Lk):
    ax1.plot(fold_train_loss, label=f"Fold {i + 1}")
ax1.legend(loc='upper right')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('cross-validation training losses')

fig2, ax2 = plt.subplots()
for i, fold_val_loss in enumerate(Lk_v):
    ax2.plot(fold_val_loss, label=f"Fold {i + 1}")
ax2.legend(loc='upper right')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.set_title('cross-validation validation losses')

fig1.savefig("./images/image1_nop.png")
fig2.savefig("./images/image2_nop.png")

df_pre = pd.read_csv('./data/predict_table.csv')
for key in results_all[0].keys():
    idx = df_pre.index[df_pre['product_pid'] == key].tolist()
    v = results_all[0][key]
    for i,results in enumerate(results_all):
        if(i!=0):
            v += results[key]
    v /= 5
    for i,id in enumerate(idx):
        df_pre.loc[id, 'apply_amt_pred'] = v[i][0]
        df_pre.loc[id, 'redeem_amt_pred'] = v[i][1]
        df_pre.loc[id, 'net_in_amt_pred'] = v[i][0] - v[i][1]
        
        
df_pre.to_csv("predict_res1.csv",index=None)

# ------------------------------------------------------------
