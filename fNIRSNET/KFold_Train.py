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
print(f"Selected Task: {task[task_id]}")

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
    batch_size = 64
    milestones = [60, 90]
    data_path = MA_data_path

    # Dynamically determine the number of subjects
    available_subs = [
        int(d) for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()
    ]
    all_sub = len(available_subs)
    print(f"Number of available subjects for MA task: {all_sub}")

    data_index = np.arange(60)
    test_index = [
        np.arange(12),
        np.arange(12, 24),
        np.arange(24, 36),
        np.arange(36, 48),
        np.arange(48, 60)
    ]

# Directory to save preprocessed data and models
root_path = os.path.join('save', task[task_id], 'KFold')
if os.path.exists(root_path):
    print(f'Path "{root_path}" already exists.')
else:
    os.makedirs(root_path)
    print(f'Created directory: {root_path}')

for n_sub in range(1, all_sub + 1):
    file_path = os.path.join(data_path, str(n_sub), f"{n_sub}_desc.mat")
    if not os.path.exists(file_path):
        print(f"Data file for subject {n_sub} not found at {file_path}. Skipping this subject.")
        continue  # Skip to the next subject

    if task_id == 0:
        sub_data, label = UFFT_subject_data(data_path, subject=n_sub)
    elif task_id == 1:
        sub_data, label = MA_subject_data(path=data_path, sub=n_sub)

    print(f"Subject {n_sub} data loaded with shape: {sub_data.shape}, Labels shape: {label.shape}")

    n_runs = 0
    for n_fold in range(5):
        n_runs += 1
        print('======================================')
        print(f'Run: {n_runs}')
        path = os.path.join(root_path, str(n_sub), str(n_runs))

        if os.path.exists(path):
            print(f'Sub path "{path}" already exists.')
        else:
            os.makedirs(path)
            print(f'Created sub-directory: {path}')

        # Training and test sets
        X_train, y_train, X_test, y_test = KFold_train_test_set(
            sub_data, label, data_index, test_index, n_fold
        )
        print(f"Fold {n_fold}:")
        print(f"  Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
        print(f"  Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

        # Save preprocessed data for future use
        train_data_path = os.path.join(path, 'train_data.npz')
        test_data_path = os.path.join(path, 'test_data.npz')
        np.savez(train_data_path, X=X_train, y=y_train)
        np.savez(test_data_path, X=X_test, y=y_test)
        print(f"Saved preprocessed training data to {train_data_path}")
        print(f"Saved preprocessed test data to {test_data_path}")

        # Load dataset
        train_set = Dataset(X_train, y_train, transform=True)
        test_set = Dataset(X_test, y_test, transform=True)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=X_test.shape[0], shuffle=False
        )
        print(f"Loaded training and testing datasets.")

        # Use GPU or CPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

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

        # Print model architecture and total parameters
        print("Model architecture:\n", net)
        total_params = sum(p.numel() for p in net.parameters())
        print(f"Total parameters: {total_params}")

        criterion = LabelSmoothing(0.1)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        lrStep = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1, last_epoch=-1
        )
        print("Initialized optimizer and learning rate scheduler.")

        # Save results
        metrics_path = os.path.join(path, 'metrics.txt')
        with open(metrics_path, 'w') as metrics:
            print(f"Opened metrics file at {metrics_path}")
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

                    # Print input shapes
                    print(f"Epoch [{epoch}/{EPOCH}], Run [{n_runs}], Batch [{i}]")
                    print(f"  Training input shape: {inputs.shape}, Labels shape: {labels.shape}")

                    outputs = net(inputs)

                    # Print output shape
                    print(f"  Output shape after forward pass: {outputs.shape}")

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
                    print(f"  Metrics written to {metrics_path}")

                    # -------------------------------------------------------------------------------------------------------------------- #
                    # Save model using TorchScript instead of torch.save
                    try:
                        # Switch to CPU before tracing to ensure compatibility
                        net_cpu = net.to('cpu')
                        net_cpu.eval()
                        # Create a dummy input with the correct shape
                        dummy_input = torch.randn(1, *X_train.shape[1:])  # Adjust batch size to 1 for tracing
                        traced_model = torch.jit.trace(net_cpu, dummy_input)
                        traced_model_path = os.path.join(path, 'model.pt')
                        traced_model.save(traced_model_path)
                        print(f'Model saved successfully at {traced_model_path}')
                    except Exception as e:
                        print(f'Error saving the model with TorchScript: {e}')

                    # Export the ONNX model using actual input data from test_loader
                    try:
                        inputs, _ = next(iter(test_loader))  # Get actual inputs from test_loader
                        inputs = inputs.to(device)

                        # Print shape of inputs used for ONNX export
                        print("Shape of input data used for ONNX export:", inputs.shape)
                        print("Exporting model to ONNX...")

                        onnx_model_path = os.path.join(path, 'model.onnx')
                        torch.onnx.export(
                            net,
                            inputs,
                            onnx_model_path,
                            input_names=['input'],
                            output_names=['output'],
                            dynamic_axes={
                                'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}
                            },
                            opset_version=10  # Ensure this matches your requirements
                        )

                        # Confirmation of export
                        print(f"ONNX model exported successfully at {onnx_model_path}")
                        print("ONNX export opset_version=10, with dynamic axes for batch size.")
                    except Exception as e:
                        print(f'Error exporting the model to ONNX: {e}')

            # Step the learning rate scheduler at the end of each epoch
            lrStep.step()
            print(f"Epoch [{epoch}/{EPOCH}] completed. Learning rate stepped.")

    print(f"Completed training for subject {n_sub}.")

print("All training runs completed.")

