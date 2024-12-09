import torch
import torch.onnx
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score
import numpy as np
import os
from LabelSmoothing import LabelSmoothing
from fNIRSNet import fNIRSNet
from dataloader import UFFT_subject_data, MA_subject_data, KFold_train_test_set, Dataset

# Select dataset through task_id
task = ['UFFT', 'MA']
task_id = 1
print(task[task_id])

# Set dataset path
UFFT_data_path = 'UFFT_data'
MA_data_path = 'MA_fNIRS_data'

if task_id == 0:
    # UFFT
    num_class = 3  # number of classes; RHT, LHT, and FT
    EPOCH = 120  # number of training epochs
    all_sub = 30  # number of subjects
    batch_size = 64  # batch size
    milestones = [60, 90]  # learning rate decay at specified epochs
    data_path = UFFT_data_path
    # Generate the index of test set
    data_index = np.arange(75)
    test_index = [
        np.arange(15),
        np.arange(15, 30),
        np.arange(30, 45),
        np.arange(45, 60),
        np.arange(60, 75)
    ]
elif task_id == 1:
    # MA
    num_class = 2  # number of classes; MA and BL
    EPOCH = 120
    all_sub = 29  # number of subjects
    batch_size = 64
    milestones = [60, 90]
    data_path = MA_data_path
    data_index = np.arange(60)
    test_index = [
        np.arange(12),
        np.arange(12, 24),
        np.arange(24, 36),
        np.arange(36, 48),
        np.arange(48, 60)
    ]

# Fix: Replace while loop with if-else to prevent infinite loop
root_path = os.path.join('save', task[task_id], 'KFold')
if os.path.exists(root_path):
    print(f'Path "{root_path}" already exists.')
else:
    os.makedirs(root_path)
    print(f'Created directory: {root_path}')

for n_sub in range(1, all_sub + 1):
    if task_id == 0:
        sub_data, label = UFFT_subject_data(data_path, subject=n_sub)
    elif task_id == 1:
        sub_data, label = MA_subject_data(path=data_path, sub=n_sub)

    n_runs = 0
    for n_fold in range(5):
        n_runs += 1
        print('======================================\nRun:', n_runs)
        path = os.path.join(root_path, str(n_sub), str(n_runs))
        
        # Fix: Replace while loop with if-else
        if os.path.exists(path):
            print(f'Sub path "{path}" already exists.')
        else:
            os.makedirs(path)
            print(f'Created sub-directory: {path}')

        # Training and test sets
        X_train, y_train, X_test, y_test = KFold_train_test_set(
            sub_data, label, data_index, test_index, n_fold
        )

        # Load dataset
        train_set = Dataset(X_train, y_train, transform=True)
        test_set = Dataset(X_test, y_test, transform=True)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=X_test.shape[0], shuffle=False
        )

        # Use GPU or CPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # fNIRSNet
        if task_id == 0:
            net = fNIRSNet(
                num_class=num_class,
                DHRConv_width=40,
                DWConv_height=40,
                num_DHRConv=4,
                num_DWConv=8
            ).to(device)
        elif task_id == 1:
            net = fNIRSNet(
                num_class=num_class,
                DHRConv_width=30,
                DWConv_height=72,
                num_DHRConv=4,
                num_DWConv=8
            ).to(device)

        criterion = LabelSmoothing(0.1)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        lrStep = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1, last_epoch=-1
        )

        # Save results
        metrics_path = os.path.join(path, 'metrics.txt')
        with open(metrics_path, 'w') as metrics:
            # -------------------------------------------------------------------------------------------------------------------- #
            # Model training
            for epoch in range(EPOCH):
                net.train()
                train_running_acc = 0
                total = 0
                loss_steps = []
                for i, data in enumerate(train_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)

                    loss = criterion(outputs, labels.long())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_steps.append(loss.item())
                    total += labels.shape[0]
                    pred = outputs.argmax(dim=1, keepdim=True)
                    train_running_acc += pred.eq(labels.view_as(pred)).sum().item()

                train_running_loss = float(np.mean(loss_steps))
                train_running_acc = 100 * train_running_acc / total
                print(f'[{n_sub}, {n_runs}, {epoch}] Train loss: {train_running_loss:.5f}')
                print(f'[{n_sub}, {n_runs}, {epoch}] Train acc: {train_running_acc:.3f}%')

                # -------------------------------------------------------------------------------------------------------------------- #
                # Model evaluation and ONNX export at the final epoch
                if epoch == EPOCH - 1:
                    net.eval()
                    test_running_acc = 0
                    total = 0
                    loss_steps = []
                    all_labels = []
                    all_preds = []
                    with torch.no_grad():
                        for data in test_loader:
                            inputs, labels = data
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            outputs = net(inputs)
                            loss = criterion(outputs, labels.long())

                            loss_steps.append(loss.item())
                            total += labels.shape[0]
                            pred = outputs.argmax(dim=1, keepdim=True)
                            test_running_acc += pred.eq(labels.view_as(pred)).sum().item()

                            all_labels.append(labels.cpu())
                            all_preds.append(pred.cpu())

                    test_running_acc = 100 * test_running_acc / total
                    test_running_loss = float(np.mean(loss_steps))
                    print(f'     [{n_sub}, {n_runs}, {epoch}] Test loss: {test_running_loss:.5f}')
                    print(f'     [{n_sub}, {n_runs}, {epoch}] Test acc: {test_running_acc:.3f}%')

                    # Concatenate all labels and predictions
                    y_label = torch.cat(all_labels).numpy()
                    y_pred = torch.cat(all_preds).numpy()

                    acc = accuracy_score(y_label, y_pred)
                    if task_id == 0:
                        # Macro mode for UFFT
                        precision = precision_score(y_label, y_pred, average='macro')
                        recall = recall_score(y_label, y_pred, average='macro')
                        f1 = f1_score(y_label, y_pred, average='macro')
                    elif task_id == 1:
                        precision = precision_score(y_label, y_pred)
                        recall = recall_score(y_label, y_pred)
                        f1 = f1_score(y_label, y_pred)
                    kappa_value = cohen_kappa_score(y_label, y_pred)
                    confusion = confusion_matrix(y_label, y_pred)
                    metrics.write(
                        f"acc={acc*100:.4f}, pre={precision*100:.4f}, rec={recall*100:.4f}, f1={f1:.4f}, kap={kappa_value:.4f}\n"
                    )
                    metrics.flush()

                    # -------------------------------------------------------------------------------------------------------------------- #
                    # Save model using TorchScript instead of torch.save
                    try:
                        # Switch to CPU before tracing to ensure compatibility
                        net_cpu = net.to('cpu')
                        net_cpu.eval()
                        # Create a dummy input with the correct shape
                        dummy_input = torch.randn(1, *X_train.shape[1:])  # Adjust batch size to 1 for tracing
                        traced_model = torch.jit.trace(net_cpu, dummy_input)
                        traced_model.save(os.path.join(path, 'model.pt'))
                        print(f'Model saved successfully at {os.path.join(path, "model.pt")}')
                    except Exception as e:
                        print(f'Error saving the model with TorchScript: {e}')

                    # Export the ONNX model using actual input data from test_loader
                    try:
                        inputs, _ = next(iter(test_loader))  # Get actual inputs from test_loader
                        inputs = inputs.to(device)
                        torch.onnx.export(
                            net,
                            inputs,
                            os.path.join(path, 'model.onnx'),
                            input_names=['input'],
                            output_names=['output'],
                            dynamic_axes={
                                'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}
                            }
                        )
                        print(f'ONNX model exported successfully at {os.path.join(path, "model.onnx")}')
                    except Exception as e:
                        print(f'Error exporting the model to ONNX: {e}')

            # Step the learning rate scheduler at the end of each epoch
            lrStep.step()

