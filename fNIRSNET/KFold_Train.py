import torch
import torch.onnx
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools

from LabelSmoothing import LabelSmoothing
from fNIRSNet import fNIRSNet
from dataloader import (
    UFFT_subject_data,
    MA_subject_data,
    KFold_train_test_set,
    Dataset
)

# Define helper functions for plotting
def plot_confusion_matrix(cm, classes, title, save_path, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc(y_true, y_score, num_classes, title, save_path):
    # Binarize the output
    y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 6))
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='Class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metric_curve(train_values, test_values, metric_name, title, save_path):
    """
    Plots a metric (e.g. accuracy or loss) curve for both train and test sets over epochs.
    """
    epochs = np.arange(1, len(train_values) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_values, label=f'Train {metric_name}', marker='o')
    plt.plot(epochs, test_values, label=f'Test {metric_name}', marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

task = ['UFFT', 'MA']
task_id = 1
print(f"Selected Task: {task[task_id]}")

# Set dataset path
UFFT_data_path = 'UFFT_data'
MA_data_path = 'MA_fNIRS_data'

if task_id == 0:
    # UFFT
    num_class = 3
    EPOCH = 120
    all_sub = 30
    batch_size = 64
    milestones = [60, 90]
    data_path = UFFT_data_path
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
    num_class = 2
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

root_path = os.path.join('save', task[task_id], 'KFold')
if os.path.exists(root_path):
    print(f'Path "{root_path}" already exists.')
else:
    os.makedirs(root_path)
    print(f'Created directory: {root_path}')

# Initialize accumulators for overall metrics
overall_labels = []
overall_preds = []
overall_scores = []
# We will also store epoch-wise metrics for overall aggregation
# After all subjects processed, we'll aggregate their metrics too
overall_train_loss_runs = []
overall_train_acc_runs = []
overall_test_loss_runs = []
overall_test_acc_runs = []

# Dictionary to store per-subject run-level metrics for later aggregation
subject_run_metrics = {}

for n_sub in range(1, all_sub + 1):
    file_path = os.path.join(data_path, str(n_sub), f"{n_sub}_desc.mat")
    if not os.path.exists(file_path):
        print(f"Data file for subject {n_sub} not found at {file_path}. Skipping this subject.")
        continue

    if task_id == 0:
        sub_data, label = UFFT_subject_data(data_path, subject=n_sub)
    elif task_id == 1:
        sub_data, label = MA_subject_data(path=data_path, sub=n_sub)

    print(f"Subject {n_sub} data loaded with shape: {sub_data.shape}, Labels shape: {label.shape}")

    subject_run_metrics[n_sub] = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    n_runs = 0
    for n_fold in range(5):
        n_runs += 1
        print('======================================')
        print(f'Run: {n_runs}')
        path = os.path.join(root_path, str(n_sub), str(n_runs))
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'Created sub-directory: {path}')

        plots_path = os.path.join(path, 'plots')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
            print(f'Created plots directory: {plots_path}')

        # Training and test sets
        X_train, y_train, X_test, y_test = KFold_train_test_set(
            sub_data, label, data_index, test_index, n_fold
        )
        print(f"Fold {n_fold}:")
        print(f"  Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
        print(f"  Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

        train_data_path = os.path.join(path, 'train_data.npz')
        test_data_path = os.path.join(path, 'test_data.npz')
        np.savez(train_data_path, X=X_train, y=y_train)
        np.savez(test_data_path, X=X_test, y=y_test)
        print(f"Saved preprocessed training data to {train_data_path}")
        print(f"Saved preprocessed test data to {test_data_path}")

        train_set = Dataset(X_train, y_train, transform=True)
        test_set = Dataset(X_test, y_test, transform=True)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=X_test.shape[0], shuffle=False
        )

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        if task_id == 0:
            net = fNIRSNet(num_class=num_class, DHRConv_width=40, DWConv_height=40, num_DHRConv=4, num_DWConv=8).to(device)
        else:
            net = fNIRSNet(num_class=num_class, DHRConv_width=30, DWConv_height=72, num_DHRConv=4, num_DWConv=8).to(device)

        print("Model architecture:\n", net)
        total_params = sum(p.numel() for p in net.parameters())
        print(f"Total parameters: {total_params}")

        criterion = LabelSmoothing(0.1)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        lrStep = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=-1)

        metrics_path = os.path.join(path, 'metrics.txt')
        print(f"Opened metrics file at {metrics_path}")

        # Lists to store metrics per epoch for this run
        train_loss_history = []
        train_acc_history = []
        test_loss_history = []
        test_acc_history = []

        for epoch in range(EPOCH):
            net.train()
            train_running_acc = 0
            total_train = 0
            loss_steps = []
            for i, data_batch in enumerate(train_loader):
                inputs, labels_batch = data_batch
                inputs = inputs.to(device)
                labels_batch = labels_batch.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels_batch.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_steps.append(loss.item())
                total_train += labels_batch.shape[0]
                pred = outputs.argmax(dim=1, keepdim=True)
                train_running_acc += pred.eq(labels_batch.view_as(pred)).sum().item()

            # Compute training metrics for this epoch
            epoch_train_loss = float(np.mean(loss_steps))
            epoch_train_acc = 100 * train_running_acc / total_train

            # Evaluate on test set after each epoch to track improvement
            net.eval()
            test_running_acc = 0
            test_loss_steps = []
            total_test = 0
            with torch.no_grad():
                for data_batch in test_loader:
                    inputs, labels_batch = data_batch
                    inputs = inputs.to(device)
                    labels_batch = labels_batch.to(device)
                    outputs = net(inputs)
                    loss_test = criterion(outputs, labels_batch.long())

                    test_loss_steps.append(loss_test.item())
                    total_test += labels_batch.shape[0]
                    pred = outputs.argmax(dim=1, keepdim=True)
                    test_running_acc += pred.eq(labels_batch.view_as(pred)).sum().item()

            epoch_test_loss = float(np.mean(test_loss_steps))
            epoch_test_acc = 100 * test_running_acc / total_test

            train_loss_history.append(epoch_train_loss)
            train_acc_history.append(epoch_train_acc)
            test_loss_history.append(epoch_test_loss)
            test_acc_history.append(epoch_test_acc)

            print(f"[{n_sub}, {n_runs}, {epoch}] Train loss: {epoch_train_loss:.5f}, Train acc: {epoch_train_acc:.3f}% | Test loss: {epoch_test_loss:.5f}, Test acc: {epoch_test_acc:.3f}%")

            # Step LR scheduler
            lrStep.step()

        # After all epochs, evaluate final metrics for confusion matrix, ROC etc.
        # (We already have final epoch outputs, but let's do a final, more detailed evaluation)
        net.eval()
        all_labels = []
        all_preds = []
        all_scores = []
        with torch.no_grad():
            for data_batch in test_loader:
                inputs, labels_batch = data_batch
                inputs = inputs.to(device)
                labels_batch = labels_batch.to(device)
                outputs = net(inputs)
                pred = outputs.argmax(dim=1, keepdim=True)

                all_labels.append(labels_batch.cpu())
                all_preds.append(pred.cpu())
                all_scores.append(outputs.cpu())

        y_label = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_score = torch.cat(all_scores).numpy()

        if task_id == 0:
            # Macro mode for UFFT
            precision = precision_score(y_label, y_pred, average='macro')
            recall = recall_score(y_label, y_pred, average='macro')
            f1 = f1_score(y_label, y_pred, average='macro')
        else:
            precision = precision_score(y_label, y_pred)
            recall = recall_score(y_label, y_pred)
            f1 = f1_score(y_label, y_pred)

        acc = accuracy_score(y_label, y_pred)
        kappa_value = cohen_kappa_score(y_label, y_pred)
        confusion = confusion_matrix(y_label, y_pred)

        with open(metrics_path, 'w') as metrics_file:
            metrics_file.write(
                f"acc={acc*100:.4f}, pre={precision*100:.4f}, rec={recall*100:.4f}, f1={f1:.4f}, kap={kappa_value:.4f}\n"
            )
        print(f"Metrics written to {metrics_path}")

        # Save model
        try:
            net_cpu = net.to('cpu')
            net_cpu.eval()
            dummy_input = torch.randn(1, *X_train.shape[1:])
            traced_model = torch.jit.trace(net_cpu, dummy_input)
            traced_model_path = os.path.join(path, 'model.pt')
            traced_model.save(traced_model_path)
            print(f'Model saved successfully at {traced_model_path}')
        except Exception as e:
            print(f'Error saving the model with TorchScript: {e}')

        # Export ONNX model
        try:
            inputs, _ = next(iter(test_loader))
            inputs = inputs.to(device)
            onnx_model_path = os.path.join(path, 'model.onnx')
            torch.onnx.export(
                net, inputs, onnx_model_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                opset_version=10
            )
            print(f"ONNX model exported successfully at {onnx_model_path}")
        except Exception as e:
            print(f'Error exporting the model to ONNX: {e}')

        # Plot ROC curve
        try:
            y_score_softmax = torch.nn.functional.softmax(torch.tensor(y_score), dim=1).numpy()
            roc_title = f'Subject {n_sub} Run {n_runs} ROC'
            roc_save_path = os.path.join(plots_path, f'ROC_Sub{n_sub}_Run{n_runs}.png')
            plot_roc(y_label, y_score_softmax, num_class, roc_title, roc_save_path)
            print(f"ROC curve saved to {roc_save_path}")
        except Exception as e:
            print(f'Error plotting ROC curve: {e}')

        # Plot Confusion Matrix
        try:
            cm_title = f'Subject {n_sub} Run {n_runs} Confusion Matrix'
            cm_save_path = os.path.join(plots_path, f'Confusion_Matrix_Sub{n_sub}_Run{n_runs}.png')
            class_names = [f"Class {i}" for i in range(num_class)]
            plot_confusion_matrix(confusion, classes=class_names, title=cm_title, save_path=cm_save_path, normalize=True)
            print(f"Confusion matrix saved to {cm_save_path}")
        except Exception as e:
            print(f'Error plotting confusion matrix: {e}')

        # Plot training & test curves (Loss and Acc)
        # Training vs Test Loss
        plot_metric_curve(train_loss_history, test_loss_history, 'loss',
                          f'Subject {n_sub} Run {n_runs} Loss Curve',
                          os.path.join(plots_path, f'Loss_Curve_Sub{n_sub}_Run{n_runs}.png'))

        # Training vs Test Accuracy
        plot_metric_curve(train_acc_history, test_acc_history, 'accuracy',
                          f'Subject {n_sub} Run {n_runs} Accuracy Curve',
                          os.path.join(plots_path, f'Accuracy_Curve_Sub{n_sub}_Run{n_runs}.png'))

        # Save run-level metrics for aggregation
        run_metrics_path = os.path.join(path, 'run_metrics.npz')
        np.savez(run_metrics_path,
                 train_loss=train_loss_history,
                 train_acc=train_acc_history,
                 test_loss=test_loss_history,
                 test_acc=test_acc_history)
        print(f"Run-level metrics saved to {run_metrics_path}")

        # Append to overall accumulators
        overall_labels.extend(y_label)
        overall_preds.extend(y_pred)
        overall_scores.append(y_score_softmax)

        # Store this run's metrics for subject-level aggregation
        subject_run_metrics[n_sub]['train_loss'].append(train_loss_history)
        subject_run_metrics[n_sub]['train_acc'].append(train_acc_history)
        subject_run_metrics[n_sub]['test_loss'].append(test_loss_history)
        subject_run_metrics[n_sub]['test_acc'].append(test_acc_history)

    # After all runs for this subject, aggregate metrics over runs and plot subject-level curves
    # Average the metrics across the 5 runs
    subject_train_loss_mean = np.mean(subject_run_metrics[n_sub]['train_loss'], axis=0)
    subject_train_acc_mean = np.mean(subject_run_metrics[n_sub]['train_acc'], axis=0)
    subject_test_loss_mean = np.mean(subject_run_metrics[n_sub]['test_loss'], axis=0)
    subject_test_acc_mean = np.mean(subject_run_metrics[n_sub]['test_acc'], axis=0)

    subject_plots_path = os.path.join(root_path, str(n_sub))
    # Subject-level aggregated loss curve
    plot_metric_curve(subject_train_loss_mean, subject_test_loss_mean, 'loss',
                      f'Subject {n_sub} Aggregated Loss Curve',
                      os.path.join(subject_plots_path, f'Subject_{n_sub}_Aggregated_Loss_Curve.png'))

    # Subject-level aggregated accuracy curve
    plot_metric_curve(subject_train_acc_mean, subject_test_acc_mean, 'accuracy',
                      f'Subject {n_sub} Aggregated Accuracy Curve',
                      os.path.join(subject_plots_path, f'Subject_{n_sub}_Aggregated_Accuracy_Curve.png'))

    # Save subject-level aggregated metrics for final overall aggregation
    subject_agg_metrics_path = os.path.join(subject_plots_path, 'subject_aggregated_metrics.npz')
    np.savez(subject_agg_metrics_path,
             train_loss=subject_train_loss_mean,
             train_acc=subject_train_acc_mean,
             test_loss=subject_test_loss_mean,
             test_acc=subject_test_acc_mean)
    print(f"Subject-level aggregated metrics saved at {subject_agg_metrics_path}")

print("Generating overall performance metrics and plots...")

if overall_labels and overall_preds:
    overall_labels = np.array(overall_labels)
    overall_preds = np.array(overall_preds)
    overall_scores = np.vstack(overall_scores)

    # Compute overall metrics
    if task_id == 0:
        overall_precision = precision_score(overall_labels, overall_preds, average='macro')
        overall_recall = recall_score(overall_labels, overall_preds, average='macro')
        overall_f1 = f1_score(overall_labels, overall_preds, average='macro')
    else:
        overall_precision = precision_score(overall_labels, overall_preds)
        overall_recall = recall_score(overall_labels, overall_preds)
        overall_f1 = f1_score(overall_labels, overall_preds)
    overall_acc = accuracy_score(overall_labels, overall_preds)
    overall_kappa = cohen_kappa_score(overall_labels, overall_preds)
    overall_confusion = confusion_matrix(overall_labels, overall_preds)

    overall_path = os.path.join(root_path, 'overall')
    if not os.path.exists(overall_path):
        os.makedirs(overall_path)
        print(f'Created overall directory: {overall_path}')

    overall_metrics_path = os.path.join(overall_path, 'overall_metrics.txt')
    with open(overall_metrics_path, 'w') as overall_metrics:
        overall_metrics.write(
            f"Overall Acc: {overall_acc*100:.4f}%, "
            f"Pre: {overall_precision*100:.4f}%, "
            f"Rec: {overall_recall*100:.4f}%, "
            f"F1: {overall_f1:.4f}, "
            f"Kappa: {overall_kappa:.4f}\n"
        )
    print(f"Overall metrics written to {overall_metrics_path}")

    # Plot overall ROC curve
    try:
        overall_roc_title = 'Overall ROC Curve'
        overall_roc_save_path = os.path.join(overall_path, 'ROC_Overall.png')
        plot_roc(overall_labels, overall_scores, num_class, overall_roc_title, overall_roc_save_path)
        print(f"Overall ROC curve saved to {overall_roc_save_path}")
    except Exception as e:
        print(f'Error plotting overall ROC curve: {e}')

    # Plot overall Confusion Matrix
    try:
        overall_cm_title = 'Overall Confusion Matrix'
        overall_cm_save_path = os.path.join(overall_path, 'Confusion_Matrix_Overall.png')
        class_names = [f"Class {i}" for i in range(num_class)]
        plot_confusion_matrix(overall_confusion, classes=class_names, title=overall_cm_title, save_path=overall_cm_save_path, normalize=True)
        print(f"Overall confusion matrix saved to {overall_cm_save_path}")
    except Exception as e:
        print(f'Error plotting overall confusion matrix: {e}')

    # Now aggregate all subject-level metrics for a final "grand average"
    all_subject_train_loss = []
    all_subject_train_acc = []
    all_subject_test_loss = []
    all_subject_test_acc = []

    # Load each subject's aggregated metrics
    for n_sub in subject_run_metrics.keys():
        subject_plots_path = os.path.join(root_path, str(n_sub))
        subject_agg_metrics_path = os.path.join(subject_plots_path, 'subject_aggregated_metrics.npz')
        if os.path.exists(subject_agg_metrics_path):
            agg_data = np.load(subject_agg_metrics_path)
            all_subject_train_loss.append(agg_data['train_loss'])
            all_subject_train_acc.append(agg_data['train_acc'])
            all_subject_test_loss.append(agg_data['test_loss'])
            all_subject_test_acc.append(agg_data['test_acc'])

    if all_subject_train_loss:
        # Compute overall averages across subjects
        grand_train_loss_mean = np.mean(all_subject_train_loss, axis=0)
        grand_train_acc_mean = np.mean(all_subject_train_acc, axis=0)
        grand_test_loss_mean = np.mean(all_subject_test_loss, axis=0)
        grand_test_acc_mean = np.mean(all_subject_test_acc, axis=0)

        # Plot the grand averaged curves
        plot_metric_curve(grand_train_loss_mean, grand_test_loss_mean, 'loss',
                          'Overall Aggregated Loss Curve',
                          os.path.join(overall_path, 'Overall_Aggregated_Loss_Curve.png'))

        plot_metric_curve(grand_train_acc_mean, grand_test_acc_mean, 'accuracy',
                          'Overall Aggregated Accuracy Curve',
                          os.path.join(overall_path, 'Overall_Aggregated_Accuracy_Curve.png'))

        # Save these grand averages
        np.savez(os.path.join(overall_path, 'overall_aggregated_metrics.npz'),
                 train_loss=grand_train_loss_mean,
                 train_acc=grand_train_acc_mean,
                 test_loss=grand_test_loss_mean,
                 test_acc=grand_test_acc_mean)
        print("Overall aggregated metrics saved and plotted.")

else:
    print("No overall data to evaluate.")

print("All training runs and evaluations completed.")

