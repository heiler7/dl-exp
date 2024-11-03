
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

df = pd.read_csv(r'./data/product_info_simple_final_train_1.csv')
grouped = df.groupby('product_pid')
samples = []
for _, group in grouped:

    product_samples = group.values

    num_samples = len(product_samples)

    if num_samples < 40:
        continue

    for i in range(num_samples - 39):
        sample = product_samples[i:i + 40, :]
        column_1 = sample[-10:, 3]
        column_2 = sample[-10:, 2]

        are_elements_same = not (np.all(column_2 != column_2[0]) or np.all(column_1 != column_1[0]))
        if are_elements_same and np.count_nonzero(column_1) == column_1.size and np.count_nonzero(column_2) == column_2.size:
            samples.append(sample)

final_array = np.array(samples)
train_data = final_array[:, :-10, 2:].astype(np.float32)
train_label = final_array[:, -10:, 2:4].astype(np.float32)

mean_values = np.mean(train_label, axis=1)
mean_broadcasted = np.tile(mean_values[:, np.newaxis, :], (1, 10, 1))
train_label = (train_label - mean_broadcasted)/(mean_broadcasted)


DATA = torch.tensor(train_data, dtype=torch.float32)
LABEL = torch.tensor(train_label, dtype=torch.float32)

batch_size = 500
learning_rate = 0.001
N = DATA.shape[0]
num_epochs = 50
k = 5
cl_splits = KFold(n_splits=k, shuffle=True, random_state=666)

L5 = []
L5_v = []
T_AUC = []
Fprs = []
Tprs = []
for fold, (train_idx, test_idx) in enumerate(cl_splits.split(np.arange(N))):
    DATA_t = DATA[train_idx]
    DATA_v = DATA[test_idx]
    LABEL_t = LABEL[train_idx]
    LABEL_v = LABEL[test_idx]

    print(DATA_v.shape, DATA_t.shape)

    dataset_train = TensorDataset(DATA_t, LABEL_t)
    dataset_val = TensorDataset(DATA_v, LABEL_v)
    print('Fold {}'.format(fold + 1))
    train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dataset_val, shuffle=True, batch_size=batch_size)


    class model1(nn.Module):

        def __init__(self,
                     fc1_size=100,
                     fc2_size=100,
                     fc3_size=100,
                     fc1_dropout=0,
                     fc2_dropout=0,
                     fc3_dropout=0,
                     num_of_classes=50):
            super(model1, self).__init__()

            self.f_model = nn.Sequential(
                nn.Linear(129, fc1_size),  # 1920
                nn.BatchNorm1d(fc1_size),
                nn.ReLU(),
                nn.Dropout(fc1_dropout),
                nn.Linear(fc1_size, fc2_size),
                nn.BatchNorm1d(fc2_size),
                nn.ReLU(),
                nn.Dropout(fc2_dropout),
                nn.Linear(fc2_size, fc3_size),
                nn.BatchNorm1d(fc3_size),
                nn.ReLU(),
                nn.Dropout(fc3_dropout),
                nn.Linear(fc3_size, 20),

            )

            self.conv_layers1 = nn.Sequential(
                nn.Conv1d(5, 1, kernel_size=1),
                nn.BatchNorm1d(1),
                nn.Dropout(fc3_dropout),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            )

            self.conv_2D = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=2),
                nn.BatchNorm2d(1),
                nn.Dropout(fc3_dropout),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
            hidden_dim = 1
            self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=50, batch_first=True,
                                # dropout=fc3_dropout,
                                bidirectional=True)



            # for name, module in self.named_modules():
            #     if isinstance(module, nn.Linear):
            #         print('Linear初始化')
            #         nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            #     if isinstance(module, nn.Conv2d):
            #         print('Conv2d初始化')
            #         nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            #     if isinstance(module, nn.Conv1d):
            #         print('Conv1d初始化')
            #         nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')



        def forward(self, x):
            with torch.no_grad():
                NN = torch.narrow(x, dim=-1, start=8, length=1)[:, -1:, ].squeeze(1).long()
                # NN = torch.reshape(NN, (NN.shape[0], NN.shape[1] * NN.shape[2]))

                # min_vals, _ = torch.min(x, dim=1, keepdim=True)
                # max_vals, _ = torch.max(x, dim=1, keepdim=True)
                # x = (x - min_vals) / (max_vals - min_vals )

                # close_prices = torch.narrow(x, dim=-1, start=1, length=1)[:, -10:, ].transpose(1, 2).squeeze(1)
                # ma_window = 5
                # ma10 = close_prices.unfold(1, ma_window, 1).mean(dim=-1)
                #
                # close_prices11 = torch.narrow(x, dim=-1, start=0, length=1)[:, -10:, ].transpose(1, 2).squeeze(1)
                # ma1110 = close_prices11.unfold(1, ma_window, 1).mean(dim=-1)

                end = torch.narrow(x, dim=-1, start=0, length=2)[:, -10:, ].squeeze(1)
                end = torch.reshape(end, (end.shape[0], end.shape[1] * end.shape[2]))
                YDD = torch.narrow(x, dim=-1, start=7, length=1)[:, -10:, ].squeeze(1)
                YDD = torch.reshape(YDD, (YDD.shape[0], YDD.shape[1] * YDD.shape[2]))
            apply = torch.narrow(x, dim=-1, start=0, length=1)[:, -30:, ].squeeze(1)
            redeem = torch.narrow(x, dim=-1, start=1, length=1)[:, -30:, ].squeeze(1)
            apply, _ = self.lstm(apply)
            redeem, _ = self.lstm(redeem)
            apply = torch.reshape(apply, (apply.shape[0], apply.shape[1] * apply.shape[2]))
            redeem = torch.reshape(redeem, (redeem.shape[0], redeem.shape[1] * redeem.shape[2]))
            YD =  torch.narrow(x, dim=-1, start=7, length=1)[:, -30:, ].squeeze(1)
            YD, _ = self.lstm(YD)
            YD = torch.reshape(YD, (YD.shape[0], YD.shape[1] * YD.shape[2]))

            x = torch.narrow(x, dim=-1, start=2, length=5)[:, -30:, ].squeeze(1)
            min_vals, _ = torch.min(x, dim=1, keepdim=True)
            max_vals, _ = torch.max(x, dim=1, keepdim=True)
            x = (x - min_vals) / (max_vals - min_vals +1)
            xx = x.unsqueeze(1)
            xx = self.conv_2D(xx)
            xx = torch.reshape(xx, (xx.shape[0], xx.shape[1] * xx.shape[2] * xx.shape[3]))

            x = x.transpose(1, 2)
            x = self.conv_layers1(x)
            out = x.transpose(1, 2)
            out2, _ = self.lstm(out)
            out2 = torch.reshape(out2, (out2.shape[0], out2.shape[1] * out2.shape[2]))
            # IN = torch.cat((xx, out2, end,NN,YD,YDD), dim=1)
            IN = torch.cat((xx, out2,NN,YD,YDD), dim=1)
            out = self.f_model(IN)
            out = torch.reshape(out, (out.shape[0], 10, 2))
            return out


    model = model1()
    model.to(device)
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    L_train = []
    L_val = []
    AUC = []
    fp = []
    tp = []
    min_validation_loss = 9999
    for epoch in range(num_epochs):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        train_running_loss = 0.0
        ACC = 0
        counter = 0
        model.train()
        for X, y in tqdm(train_loader):
            optimizer.zero_grad()
            counter += 1
            output = model(X.to(device))
            loss = criterion(output, y.to(device))
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        TL = train_running_loss / counter
        L_train.append(TL)
        model.eval()
        PREDICT = []
        TRUE = []
        ACC = 0
        counter = 0
        with torch.no_grad():
            current_test_loss = 0.0

            for XX, Z in tqdm(val_loader):
                counter += 1
                output = model(XX.to(device))
                loss = criterion(output, Z.to(device))
                current_test_loss += loss.item()
                PREDICT.extend(output.cpu().numpy())
                TRUE.extend(Z.cpu().numpy())
            T_loss = current_test_loss / counter
            L_val.append(T_loss)
            if min_validation_loss > T_loss:
                min_validation_loss = T_loss
                best_epoch = epoch
                print('Min validation_loss ' + str(min_validation_loss) + ' in epoch ' + str(best_epoch))
                torch.save(model.state_dict(), fr"weight/model_1_{fold}.pt")
        PP = np.array(PREDICT)
        TT = np.array(TRUE)
        flattened_array1 = PP.flatten()
        flattened_array2 = TT.flatten()
        correlation_matrix = np.corrcoef(flattened_array1, flattened_array2)
        correlation_value = correlation_matrix[0, 1]

        print("Train loss: ", TL, "Val loss: ", T_loss,'correlation_value',correlation_value)
        # AUC.append()
    L5.append(L_train)
    L5_v.append(L_val)
    # T_AUC.append(AUC)

fig1, ax1 = plt.subplots()
for i, fold_train_loss in enumerate(L5):
    ax1.plot(fold_train_loss, label=f"Fold {i + 1}")
ax1.legend(loc='upper right')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('5-fold cross-validation training losses')


fig2, ax2 = plt.subplots()
for i, fold_val_loss in enumerate(L5_v):
    ax2.plot(fold_val_loss, label=f"Fold {i + 1}")
ax2.legend(loc='upper right')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.set_title('5-fold cross-validation validation losses')


fig1.savefig("./images/image1_nop.png")
fig2.savefig("./images/image2_nop.png")

