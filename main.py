import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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

# class model1(nn.Module):
#     def __init__(self,
#                     256=100,
#                     256=100,
#                     fc3_size=100,
#                     fc1_dropout=0,
#                     fc2_dropout=0,
#                     fc3_dropout=0,
#                     num_of_classes=50):
#         super(model1, self).__init__()

#         self.f_model = nn.Sequential(
#             nn.Linear(259+10, 256),  # 1920
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(fc1_dropout),
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(fc2_dropout),
#             nn.Linear(256, fc3_size),
#             nn.BatchNorm1d(fc3_size),
#             nn.ReLU(),
#             nn.Dropout(fc3_dropout),
#             nn.Linear(fc3_size, 20),

#         )

#         self.conv_layers1 = nn.Sequential(
#             nn.Conv1d(5, 1, kernel_size=1),
#             nn.BatchNorm1d(1),
#             nn.Dropout(fc3_dropout),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#         )

#         self.conv_2D = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=2),
#             nn.BatchNorm2d(1),
#             nn.Dropout(fc3_dropout),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         hidden_dim = 1
#         self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=50, batch_first=True,
#                             # dropout=fc3_dropout,
#                             bidirectional=True)



#         # for name, module in self.named_modules():
#         #     if isinstance(module, nn.Linear):
#         #         print('Linear初始化')
#         #         nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
#         #     if isinstance(module, nn.Conv2d):
#         #         print('Conv2d初始化')
#         #         nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
#         #     if isinstance(module, nn.Conv1d):
#         #         print('Conv1d初始化')
#         #         nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')



#     def forward(self, x):
#         with torch.no_grad():
#             NN = torch.narrow(x, dim=-1, start=8, length=1)[:, -1:, ].squeeze(1).long()
#             # NN = torch.reshape(NN, (NN.shape[0], NN.shape[1] * NN.shape[2]))

#             # min_vals, _ = torch.min(x, dim=1, keepdim=True)
#             # max_vals, _ = torch.max(x, dim=1, keepdim=True)
#             # x = (x - min_vals) / (max_vals - min_vals )

#             # close_prices = torch.narrow(x, dim=-1, start=1, length=1)[:, -10:, ].transpose(1, 2).squeeze(1)
#             # ma_window = 5
#             # ma10 = close_prices.unfold(1, ma_window, 1).mean(dim=-1)
#             #
#             # close_prices11 = torch.narrow(x, dim=-1, start=0, length=1)[:, -10:, ].transpose(1, 2).squeeze(1)
#             # ma1110 = close_prices11.unfold(1, ma_window, 1).mean(dim=-1)

#             end = torch.narrow(x, dim=-1, start=0, length=2)[:, -10:, ].squeeze(1)
#             end = torch.reshape(end, (end.shape[0], end.shape[1] * end.shape[2]))
#             YDD = torch.narrow(x, dim=-1, start=7, length=1)[:, -10:, ].squeeze(1)
#             YDD = torch.reshape(YDD, (YDD.shape[0], YDD.shape[1] * YDD.shape[2]))
#         apply = torch.narrow(x, dim=-1, start=0, length=1)[:, -30:, ].squeeze(1)
#         redeem = torch.narrow(x, dim=-1, start=1, length=1)[:, -30:, ].squeeze(1)
#         apply, _ = self.lstm(apply)
#         redeem, _ = self.lstm(redeem)
#         apply = torch.reshape(apply, (apply.shape[0], apply.shape[1] * apply.shape[2]))
#         redeem = torch.reshape(redeem, (redeem.shape[0], redeem.shape[1] * redeem.shape[2]))
#         YD =  torch.narrow(x, dim=-1, start=7, length=1)[:, -30:, ].squeeze(1)
#         YD, _ = self.lstm(YD)
#         YD = torch.reshape(YD, (YD.shape[0], YD.shape[1] * YD.shape[2]))

#         x = torch.narrow(x, dim=-1, start=2, length=5)[:, -30:, ].squeeze(1)
#         min_vals, _ = torch.min(x, dim=1, keepdim=True)
#         max_vals, _ = torch.max(x, dim=1, keepdim=True)
#         x = (x - min_vals) / (max_vals - min_vals +1)
#         xx = x.unsqueeze(1)
#         xx = self.conv_2D(xx)
#         xx = torch.reshape(xx, (xx.shape[0], xx.shape[1] * xx.shape[2] * xx.shape[3]))

#         x = x.transpose(1, 2)
#         x = self.conv_layers1(x)
#         out = x.transpose(1, 2)
#         out2, _ = self.lstm(out)
#         out2 = torch.reshape(out2, (out2.shape[0], out2.shape[1] * out2.shape[2]))

#         IN = torch.cat((xx, out2,  apply, redeem,end,NN,YD,YDD), dim=1)
#         out = self.f_model(IN)
#         out = torch.reshape(out, (out.shape[0], 10, 2))
#         return out

# 定义LSTM模型

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


class TransformerTimeSeries(nn.Module):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        channels=512,
        dropout=0.1,
    ):
        super().__init__()
        self.dropout = dropout
        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.input_projection = nn.Linear(n_encoder_inputs, channels)
        self.output_projection = nn.Linear(n_decoder_inputs, channels)

        self.fc = nn.Linear(channels, 2)
        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src
    
    def gen_trg_mask(self,length, device):
        mask = torch.tril(torch.ones(length, length, device=device)) == 1

        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
    
    def decode_trg(self, trg, memory):

        trg_start = self.output_projection(trg).permute(1, 0, 2)
        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start
        trg_mask = self.gen_trg_mask(out_sequence_len, trg.device)

        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start
        out = out.permute(1, 0, 2)
        out = self.fc(out)

        return out
    
    def forward(self, x,target):
        apply = torch.narrow(x, dim=-1, start=2, length=1)[:, :, ].squeeze(1)
        redeem = torch.narrow(x, dim=-1, start=3, length=1)[:, :, ].squeeze(1)
        # target = torch.narrow(x, dim=-1, start=2, length=2)[:, :, ].squeeze(1)
        src = x
        src = self.encode_src(src)
        output = self.decode_trg(trg=target, memory=src)
        return output

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
            results[i] = np.array(output.cpu()).astype(np.float32)

    return results
    
def predict_trans(path,predict_input,model):
    results = {}
    df = pd.read_csv(path)
    print(df.columns)
    products = df['product_pid'].unique()
    model.eval()
    with torch.no_grad():
        for i in products:
            data = predict_input[i]
            data = np.array(data).astype(np.float32)
            data = torch.tensor(data, dtype=torch.float32)
            data = data.reshape(1,data.shape[0],data.shape[1])
            target1 = torch.narrow(data, dim=-1, start=2, length=2)[0,-1,:]
            target1 = target1.reshape(1,1,-1).to(device)
            # output = model(data.to(device))
            for j in range(10):
                output = model(data.to(device),target1)
                output = output[:,-1:,:]
                target1 = torch.cat([target1,output],dim=1)
            
            # test
            # target1 = target1[:,1:,:]
            # output1 = model(data.to(device),target1).squeeze(0)
            # results[i] = np.array(output1.cpu())
            
            # before
            target1 = target1[:,1:,:].squeeze(0)
            results[i] = np.array(target1.cpu()).astype(np.float32)

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
train_data = final_array[: , :-10 , :].astype(np.float32)
train_label = final_array[: , -10: , 2:4].astype(np.float32)

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
cl_splits = KFold(n_splits=k, shuffle=False)
splits = cl_splits.split(np.arange(N))

for i, (train_idx, test_idx) in enumerate(splits):
    print('Fold {}'.format(i + 1))
    data_train = data[train_idx.tolist()]
    data_val = data[test_idx.tolist()]
    label_train = labels[train_idx.tolist()]
    label_val = labels[test_idx.tolist()]

    print(data_val.shape, data_train.shape)

    dataset_train = TensorDataset(data_train, label_train)
    dataset_val = TensorDataset(data_val, label_val)
    train_loader = DataLoader(dataset_train, shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(dataset_val, shuffle=False, batch_size=batch_size)
    
    model = TransformerTimeSeries(input_dim, target_input_dim)
    
    # model = LSTM(input_dim=1, hidden_dim=2, output_dim=20)
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
            # output = model(data_t.to(device))
            
            # Transformers:
            target = torch.cat([data_t[:,-1:,2:4],y],dim=1).to(device)
            output = model(data_t.to(device),target)
            output = output[:,:-1,:]
            
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
        with torch.no_grad():
            current_test_loss = 0.0

            for data_v, y in tqdm(val_loader):
                counter += 1
                # output = model(data_v.to(device),y.to(device))
                
                # Transformers:
                target = torch.cat([data_v[:,-1:,2:4],y],dim=1).to(device)
                output = model(data_v.to(device),target)
                output = output[:,:-1,:]
                
                loss = criterion(output, y.to(device))
                current_test_loss += loss.item()
                predict_res.extend(output.cpu().numpy())
                labels_true.extend(y.cpu().numpy())
            T_loss = current_test_loss / counter
            L_val.append(T_loss)
            if min_validation_loss > T_loss:
                min_validation_loss = T_loss
                best_epoch = epoch
                print('Min validation_loss ' + str(min_validation_loss) + ' in epoch ' + str(best_epoch))
                torch.save(model.state_dict(), fr"weight/model_1_{i}.pt")
        predict_res = np.array(predict_res)
        labels_true = np.array(labels_true)
        flattened_array1 = predict_res.flatten()
        flattened_array2 = labels_true.flatten()
        correlation_matrix = np.corrcoef(flattened_array1, flattened_array2)
        correlation_value = correlation_matrix[0, 1]

        print("Train loss: ", TL, "Val loss: ", T_loss,'correlation_value',correlation_value)
        
    Lk.append(L_train)
    Lk_v.append(L_val)
    
    # results1 = predict(predict_path,predict_input,model)
    # transformer:
    # results1 = predict_trans(predict_path,predict_input,model)
    results1 = predict_trans(predict_path,predict_trans_input,model)
    results_all.append(results1)

fig1, ax1 = plt.subplots()
for i, fold_train_loss in enumerate(Lk):
    ax1.plot(fold_train_loss, label=f"Fold {i + 1}")
ax1.legend(loc='upper right')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('5-fold cross-validation training losses')

fig2, ax2 = plt.subplots()
for i, fold_val_loss in enumerate(Lk_v):
    ax2.plot(fold_val_loss, label=f"Fold {i + 1}")
ax2.legend(loc='upper right')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.set_title('5-fold cross-validation validation losses')

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
        
        
df_pre.to_csv("predict_res2.csv",index=None)

# ------------------------------------------------------------
