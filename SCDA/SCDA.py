import argparse
import sys
import time
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from matplotlib import pyplot as plt
from pathlib import Path

# specify a seed for repeating the exact dataset splits
SEED = 28213
torch.manual_seed(SEED)
np.random.seed(seed=SEED)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='''
        Sparse Convolutional Denoising Autoencoder
        ported from https://github.com/work-hard-play-harder/SCDA
    '''
)
parser.add_argument(
    'input', help='path to the dataset (haplotypes)'
)
parser.add_argument(
    'mafs_path',
    help='path to the mafs extracted with the scripts in this dir (maf_from_xxx)'
)
parser.add_argument(
    '--results_path',
    help='path where the results should be saved'
)
parser.add_argument(
    '--plot', default=False, action='store_true',
    help='plot results after training'
)
parser.add_argument(
    '--missing_perc', type=float, default=0.95,
    help='percentage of missing SNPs for the randomly generated input'
)
parser.add_argument(
    '--hammerblade', default=False, action='store_true',
    help='run SCDA on HammerBlade'
)
parser.add_argument(
    '--lr', type=float, default=0.0001,
    help='learning rate'
)
parser.add_argument(
    '--max_epochs', type=int, default=10,
    help='max training epochs'
)
parser.add_argument(
    '--filter_size', type=int, default=5,
    help='convolution filter size'
)
parser.add_argument(
    '--batch_size', type=int, default=4,
    help='batch size'
)
parser.add_argument(
    '--channels', type=int, nargs=3, default=[32, 64, 128],
    help='number of channels for the hidden layers'
)
parser.add_argument(
    '--dropout_amount', type=float, default=0.25,
    help='dropout intensity in hidden layers'
)


#--------------------------------------------------------------#
# Sparse Convolutional Denoising Autoencoder                   #
# ported from https://github.com/work-hard-play-harder/SCDA    #
#--------------------------------------------------------------#

class SCDA(nn.Module):
    def __init__(
            self, in_channels, hidden_channels, dropout_amount, filter_size
    ):
        super(SCDA, self).__init__()
        second_channels, third_channels, fourth_channels = hidden_channels
        pooling_factor = 2 # left fixed because other values caused crashes
        # TODO: check if setting this to False helps with unbalanced classes
        use_bias = True
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels, second_channels, kernel_size=filter_size,
                bias=use_bias, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(pooling_factor),
            nn.Dropout(dropout_amount),
            nn.Conv1d(
                second_channels, third_channels, kernel_size=filter_size,
                bias=use_bias, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(pooling_factor),
            nn.Dropout(dropout_amount)
        )
        # bridge
        self.bridge = nn.Conv1d(
            third_channels, fourth_channels, kernel_size=filter_size,
            bias=use_bias, padding=2
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(
                fourth_channels, third_channels, kernel_size=filter_size,
                bias=use_bias, padding=2
            ),
            nn.ReLU(),
            nn.Upsample(scale_factor=pooling_factor),
            nn.Dropout(dropout_amount),
            nn.Conv1d(
                third_channels, second_channels, kernel_size=filter_size,
                bias=use_bias, padding=2
            ),
            nn.ReLU(),
            nn.Upsample(scale_factor=pooling_factor),
            nn.Dropout(dropout_amount),
            nn.Conv1d(
                second_channels, in_channels, kernel_size=filter_size,
                bias=use_bias, padding=2
            ),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bridge(x)
        x = self.decoder(x)
        return x

#--------------------------------------------------------------#
# Fully Connected Denoising Autoencoder v1                     #
#--------------------------------------------------------------#

class FCDAv1(nn.Module):
    def __init__(
            self, in_feats, hidden_feats
    ):
        super(FCDAv1, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_feats, in_feats),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#--------------------------------------------------------------#
# Fully Connected Denoising Autoencoder v2                     #
#--------------------------------------------------------------#

class FCDAv2(nn.Module):
    def __init__(
            self, num_classes, in_feats, hidden_feats
    ):
        super(FCDAv2, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_classes * in_feats, hidden_feats),
            nn.ReLU(),
        )
        # decoder
        self.decoder = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(hidden_feats, num_classes * in_feats),
            nn.Unflatten(-1, [num_classes, in_feats])
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#--------------------------------------------------------------#
# Fully Connected Denoising Autoencoder v3                     #
#--------------------------------------------------------------#

class FCDAv3(nn.Module):
    def __init__(
            self, num_classes, in_feats, hidden_feats
    ):
        second_feats, third_feats = hidden_feats
        # dropout_amount = 0.25
        super(FCDAv3, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_classes * in_feats, second_feats),
            nn.ReLU(),
            # nn.Dropout(dropout_amount),
            nn.Linear(second_feats, third_feats),
            nn.ReLU(),
            # nn.Dropout(dropout_amount),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(third_feats, second_feats),
            nn.ReLU(),
            # nn.Dropout(dropout_amount),
            nn.Linear(second_feats, num_classes * in_feats),
            nn.Unflatten(-1, [num_classes, in_feats])
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#-------------------------------------------------------------
# Helpers
#-------------------------------------------------------------

# convert dataframe to torch tensor
def dataframe_to_tensor(df):
    return torch.from_numpy(df.values)

# compute loss to show during training
# use reduction=none to keep individual feat (SNP) losses separate and
# aggregate by MAF later
# TODO: add weight for unbalanced classes?
criterion_for_display = nn.CrossEntropyLoss(reduction='none')
def batch_loss_for_display(z, y):
    loss_for_display = criterion_for_display(z, y).detach()
    loss_for_display.requires_grad = False
    cpu_loss_for_display = loss_for_display.cpu().numpy()
    return np.mean(cpu_loss_for_display, axis=0)

# compute accuracy for each sample in the batch
def batch_accuracy(z, y):
    _, predicted_class = torch.max(z, 1) # get index of the class with max score
    true_positives = predicted_class == y
    # for each sample in the batch, get correct SNPs / total SNPs
    num_snps = y.size(1)
    batch_acc = true_positives.sum(dim=1) / num_snps
    cpu_batch_acc = batch_acc.cpu().numpy()
    return cpu_batch_acc

# compute and accumulate confusion matrix values
# by individual SNP rather than by sample
def get_confusion_matrix_by_snp(z, y, accumulator):
    batch_size, num_variants, num_snps = z.size()
    _, predicted_class = torch.max(z, 1) # get index of the class with max score

    # for each variant other than 0 (missing), get the TP, TN, FP and FN.
    for variant in range(1, num_variants):
        # first initialize if not defined
        if variant not in accumulator:
            accumulator[variant] = {
                'true_positives': np.zeros(num_snps),
                'false_positives': np.zeros(num_snps),
                'true_negatives': np.zeros(num_snps),
                'false_negatives': np.zeros(num_snps),
            }
        target_postv_for_variant = (y == variant).cpu()
        predicted_postv_for_variant = (predicted_class == variant).cpu()
        target_negtv_for_variant = np.logical_not(target_postv_for_variant)
        predicted_negtv_for_variant = np.logical_not(predicted_postv_for_variant)
        # sum within batch and accumulate
        accumulator[variant]['true_positives'] += np.logical_and(
            predicted_postv_for_variant, target_postv_for_variant
        ).sum(dim=0).numpy()
        accumulator[variant]['false_positives'] += np.logical_and(
            predicted_postv_for_variant, target_negtv_for_variant
        ).sum(dim=0).numpy()
        accumulator[variant]['true_negatives'] += np.logical_and(
            predicted_negtv_for_variant, target_negtv_for_variant
        ).sum(dim=0).numpy()
        accumulator[variant]['false_negatives'] += np.logical_and(
            predicted_negtv_for_variant, target_postv_for_variant
        ).sum(dim=0).numpy()

# for each class, compute Matthews correlation coefficient using the existing
# confusion matrix and store in the metrics dict
def compute_metrics(conf_matrix_by_snp):
    metrics = {
        metric: {} for metric in [
            'support', 'mcc', 'accuracy',
            'precision', 'recall', 'f1-score'
        ]
    }
    for variant in conf_matrix_by_snp:
        conf_matrix = conf_matrix_by_snp[variant]

        positives = (
            conf_matrix['true_positives'] + conf_matrix['false_negatives']
        )
        negatives = (
            conf_matrix['true_negatives'] + conf_matrix['false_positives']
        )
        predicted_positives = (
            conf_matrix['true_positives'] + conf_matrix['false_positives']
        )
        predicted_negatives = (
            conf_matrix['true_negatives'] + conf_matrix['false_negatives']
        )
        total = (
            positives + negatives
        )

        metrics['support'][variant] = positives

        mcc_numerator = (
            (conf_matrix['true_positives'] * conf_matrix['true_negatives'])
            - (conf_matrix['false_positives'] * conf_matrix['false_negatives'])
        )
        mcc_denominator = np.sqrt(
            predicted_positives
            * positives
            * negatives
            * predicted_negatives
        )
        metrics['mcc'][variant] = np.divide(
            mcc_numerator,
            mcc_denominator,
            out=np.zeros_like(mcc_numerator),
            where=mcc_denominator!=0
        )

        metrics['precision'][variant] = np.divide(
            conf_matrix['true_positives'],
            predicted_positives,
            out=np.zeros_like(predicted_positives),
            where=predicted_positives!=0
        )

        metrics['recall'][variant] = np.divide(
            conf_matrix['true_positives'],
            positives,
            out=np.zeros_like(positives),
            where=positives!=0
        )

        metrics['accuracy'][variant] = np.divide(
            conf_matrix['true_positives'] + conf_matrix['true_negatives'],
            total,
            out=np.zeros_like(total),
            where=total!=0
        )

        f1_numerator = (
            2 * metrics['precision'][variant] * metrics['recall'][variant]
        )
        f1_denominator = (
            metrics['precision'][variant] + metrics['recall'][variant]
        )
        metrics['f1-score'][variant] = np.divide(
            f1_numerator,
            f1_denominator,
            out=np.zeros_like(f1_denominator),
            where=f1_denominator!=0
        )
    return metrics

# print average loss for each MAF category
def get_loss_by_maf(
        subset_name, epoch, losses, accuracies, metrics_by_snp,
        mafs, snps
    ):
    VERY_LOW = '[0-0.5%)'
    LOW = '[0.5-5%)'
    HIGH = '[5-50%)'
    buckets = {bucket: [] for bucket in (VERY_LOW, LOW, HIGH)}

    # initialize loss accumulator for each MAF category
    loss_by_maf = {
        bucket: 0.0 for bucket in buckets
    }

    avg_loss_by_feat = np.mean(losses, axis=0)

    # initialize metrics accumulator for each MAF category
    metrics_by_maf = {}
    for bucket in buckets:
        metrics_by_maf[bucket] = {
            metric: {
                'macro': 0.0,
                'micro': 0.0,
            } for metric in metrics_by_snp if metric != 'support'
        }

    for index, snp_id in enumerate(snps):
        maf = mafs[1][snp_id]
        if maf < 0.005:
            bucket = VERY_LOW
        elif maf < 0.05:
            bucket = LOW
        else:
            bucket = HIGH
        # TODO: see how to remove SNPs with a single variant due to dataset splitting
        buckets[bucket] += [index]
        loss_by_maf[bucket] += avg_loss_by_feat[index]
        for metric in metrics_by_maf[bucket]:
            scores = []
            supports = []
            for variant in metrics_by_snp[metric]:
                if metrics_by_snp['support'][variant][index] > 0:
                    scores += [metrics_by_snp[metric][variant][index]]
                    supports += [metrics_by_snp['support'][variant][index]]
            macro_average = np.mean(scores)
            micro_average = np.dot(scores, supports) / sum(supports)
            metrics_by_maf[bucket][metric]['macro'] += macro_average
            metrics_by_maf[bucket][metric]['micro'] += micro_average

    snps_by_category = {
        bucket: len(buckets[bucket]) for bucket in buckets
    }

    # divide by the number of SNPs in the bucket to get the average
    for bucket in buckets:
        snps_in_category = snps_by_category[bucket]

        if snps_in_category > 0:
            loss_by_maf[bucket] /= snps_in_category

            for metric in metrics_by_maf[bucket]:
                metrics_by_maf[bucket][metric]['macro'] /= snps_in_category
                metrics_by_maf[bucket][metric]['micro'] /= snps_in_category

    print(
        f'Epoch {epoch}; {subset_name} set\n'
        f'Loss={np.mean(avg_loss_by_feat):.6f}\n'
        f'Average accuracy={np.mean(accuracies):.6f}\n'
        f'SNPs by MAF: {json.dumps(snps_by_category, indent=2)}\n'
        f'Loss by MAF (avg within bucket): {json.dumps(loss_by_maf, indent=2)}\n'
        f'Metrics by MAF (avg within bucket): {json.dumps(metrics_by_maf, indent=2)}\n'
    )

    return loss_by_maf, metrics_by_maf

def plot_metric(
        histories, dataset, metric, average_type='macro', results_path=None
    ):
    history = histories[dataset]
    # split by MAF
    metric_history = {
        bucket: [] for bucket in history[0]
    }
    # extract score for all epochs for the requested metric
    for epoch_metrics in history:
        for bucket in epoch_metrics:
            metric_history[bucket].append(
                epoch_metrics[bucket][metric][average_type]
            )
    plt.figure()
    for bucket in metric_history:
        line = metric_history[bucket]
        plt.plot(range(len(history)), line, label=bucket)

    plt.title(f'{dataset} {metric} history')
    plt.legend()
    if results_path:
        plt.savefig(f'{results_path}/{dataset}-{metric}.svg')
    plt.show(block=True)
    #plt.savefig(f'Results/{dataset}_{metric}_history.png')

# this approach is faster when the percentage of missing SNPs in the
# generated noisy input is high
class HighMissingDataset(torch.utils.data.Dataset):
    def __init__(self, one_hot_targets, int_targets, missing_perc):
        self.one_hot_targets = one_hot_targets
        self.int_targets = int_targets
        self.missing_perc = missing_perc
        self.zero_pad = nn.ZeroPad2d((0, 0, 0, one_hot_targets.size(1) - 1))

    def __len__(self):
        return len(self.one_hot_targets)

    def __getitem__(self, index):
        int_target = self.int_targets[index]
        one_hot_target = self.one_hot_targets[index]
        # create tensor with the same size but with all one-hot feats set to 0
        noisy_input = self.zero_pad(torch.ones((1, one_hot_target.size(-1))))
        non_missing_amount = int((1 - self.missing_perc) * noisy_input.size(-1))
        # shuffle indexes to pick non-missing feats (SNPs) at random
        snp_indexes = list(range(noisy_input.size(-1)))
        np.random.shuffle(snp_indexes)
        # insert the right amount of non-missing feats in the generated tensor
        for i in list(range(non_missing_amount)):
            snp_index = snp_indexes.pop()
            noisy_input[0][snp_index] = 0.0
            for c in range(1, noisy_input.size(0)):
                noisy_input[c][snp_index] = one_hot_target[c][snp_index]
        return noisy_input, int_target

# this approach is faster when the percentage of missing SNPs in the
# generated noisy input is low
class LowMissingDataset(torch.utils.data.Dataset):
    def __init__(self, one_hot_targets, int_targets, missing_perc):
        self.one_hot_targets = one_hot_targets
        self.int_targets = int_targets
        self.missing_perc = missing_perc

    def __len__(self):
        return len(self.one_hot_targets)

    def __getitem__(self, index):
        int_target = self.int_targets[index]
        # start from a copy of the target
        noisy_input = torch.clone(self.one_hot_targets[index])
        missing_amount = int(self.missing_perc * noisy_input.size(-1))
        # Never use the line below to generate the indexes:
        # np.random.randint(noisy_input.size(-1), size=missing_amount)
        # it can create many duplicate indexes and produce noisy inputs
        # with a different amount of missing SNPs than expected.
        # Instead, shuffle the indexes to pick missing feats (SNPs) at random
        snp_indexes = list(range(noisy_input.size(-1)))
        np.random.shuffle(snp_indexes)
        # set the right amount of feats to 0 (missing) in the copy
        for i in list(range(missing_amount)):
            snp_index = snp_indexes.pop()
            noisy_input[0][snp_index] = 1.0
            for c in range(1, noisy_input.size(0)):
                noisy_input[c][snp_index] = 0.0
        return noisy_input, int_target

# Find SNPs that have at least 2 variants in the training set;
# the invalid ones (with a single variant) arise due to the dataset splitting
def find_valid_snps(one_hot_data):
    valid_indexes = []
    total_samples, total_snps, total_classes = one_hot_data.size()
    samples_per_variant = one_hot_data.sum(dim=0)
    for index in range(total_snps):
        valid = True
        for in_variant in samples_per_variant[index]:
            if in_variant == total_samples:
                valid = False
                break
        if valid:
            valid_indexes += [index]
    return valid_indexes

#-------------------------------------------------------------
# Main
#-------------------------------------------------------------

if __name__ == '__main__':
    start_time = time.time()

    args = parser.parse_args()

    input_path = args.input
    whole_dataframe = pd.read_csv(input_path, sep='\t', index_col=0)
    mafs = pd.read_csv(args.mafs_path, header=None, index_col=0)

    # make it fit for HammerBlade
    if args.hammerblade:
        whole_dataframe = whole_dataframe.iloc[:,:28160]

    print(f'{input_path} data loaded. Shape = {str(whole_dataframe.shape)}')
    print('Sample data:')
    print(whole_dataframe.head())

    # TODO: chekc if the stratify param can be used for class imbalance
    # train / validation / test split
    learning_df, test_df = train_test_split(
        whole_dataframe, test_size=0.2, random_state=SEED
    ) # test_df might be used in the future
    train_df, valid_df = train_test_split(
        learning_df, test_size=0.2, random_state=SEED
    )

    training_tensor = dataframe_to_tensor(train_df)
    training_one_hot = nn.functional.one_hot(training_tensor).float()

    # this is only necessary for the training set because during
    '''valid_indexes = find_valid_snps(training_one_hot)
    if len(valid_indexes) == 0:
        sys.exit('No valid SNPs found in the training set')
    print(f'Number of invalid SNPs: {training_one_hot.size(1) - len(valid_indexes)}')
    snp_ids = whole_dataframe.columns.values[valid_indexes]

    training_tensor = training_tensor[:, valid_indexes]
    training_one_hot = training_one_hot[:, valid_indexes, :]

    validation_tensor = dataframe_to_tensor(valid_df)[:, valid_indexes]
    validation_one_hot = nn.functional.one_hot(validation_tensor).float()'''

    snp_ids = whole_dataframe.columns.values
    
    validation_tensor = dataframe_to_tensor(valid_df)
    validation_one_hot = nn.functional.one_hot(validation_tensor).float()

    # TODO: extract valid SNPs from the test set too

    print(f'training set shape: {str(training_one_hot.size())}')
    print(f'validation set shape: {str(validation_one_hot.size())}')

    num_samples, num_features, num_classes = training_one_hot.size()

    # init FCDAv1 model
    # model = FCDAv1(
    #     num_features, 60
    # )
    # init FCDAv2 model
    # model = FCDAv2(
    #     num_classes, num_features, 100
    # )
    # init FCDAv3 model
    # model = FCDAv3(
    #     num_classes, num_features, [300, 20]
    # )
    # # init SCDA model
    model = SCDA(
        num_classes, args.channels, args.dropout_amount, args.filter_size
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.hammerblade:
        device = torch.device('hammerblade')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print()
    print(model)
    print(f'Number of parameters: {num_params}')
    print()

    results_path = args.results_path
    if results_path:
        print(f'Results will be written to {results_path}')
        Path(results_path).mkdir(parents=True, exist_ok=True)
        with open(f'{results_path}/model-info.txt', 'w') as model_file:
            model_file.write(str(model))
            model_file.write(f'\nNumber of parameters: {num_params}\n')
            model_file.write(str(args))

    # convert channel-last to channel-first
    # permute required because conv1d expects input of size
    # (batch_size, channels, instance_length)
    print(f'channel-last one hot tensor: {str(training_one_hot.size())}')
    training_one_hot = training_one_hot.permute(0,2,1).contiguous()
    validation_one_hot = validation_one_hot.permute(0,2,1).contiguous()
    print(f'channel-first one hot tensor: {str(training_one_hot.size())}\n')

    # create data loaders
    if args.missing_perc > 0.5:
        DataGenerator = HighMissingDataset
    else:
        DataGenerator = LowMissingDataset

    training_set = DataGenerator(
        training_one_hot,
        training_tensor,
        missing_perc=args.missing_perc
    )
    validation_set = DataGenerator(
        validation_one_hot,
        validation_tensor,
        missing_perc=args.missing_perc
    )

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0
    }
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    # train the model
    criterion = nn.CrossEntropyLoss() # TODO: add weight for unbalanced classes?
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    time_before_training = time.time()
    history = {
        'training': [],
        'validation': []
    }
    for epoch in range(args.max_epochs):
        model.train()
        losses_for_display = []
        accuracies = []
        conf_matrix_by_snp = {}
        for batch_idx, (x, y) in tqdm(
                enumerate(training_generator), total=len(training_generator)
        ):
            x, y = x.contiguous(), y.contiguous()

            if args.hammerblade:
                x, y = x.hammerblade(), y.hammerblade()
            else:
                x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            z = model(x)

            losses_for_display.append(batch_loss_for_display(z, y))
            get_confusion_matrix_by_snp(z, y, conf_matrix_by_snp)
            accuracies = np.concatenate((accuracies, batch_accuracy(z, y)))

            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

        metrics_by_snp = compute_metrics(conf_matrix_by_snp)
        loss_by_maf, metrics_by_maf = get_loss_by_maf(
            'training', epoch, losses_for_display, accuracies,
            metrics_by_snp, mafs, snp_ids
        )
        history['training'].append(metrics_by_maf)

        # validation
        model.eval()
        losses_for_display = []
        accuracies = []
        conf_matrix_by_snp = {}
        for batch_idx, (x, y) in tqdm(
            enumerate(validation_generator), total=len(validation_generator)
        ):
            x, y = x.contiguous(), y.contiguous()
            x, y = x.to(device), y.to(device)

            z = model(x)

            losses_for_display.append(batch_loss_for_display(z, y))
            get_confusion_matrix_by_snp(z, y, conf_matrix_by_snp)
            accuracies = np.concatenate((accuracies, batch_accuracy(z, y)))

        metrics_by_snp = compute_metrics(conf_matrix_by_snp)
        loss_by_maf, metrics_by_maf = get_loss_by_maf(
            'validation', epoch, losses_for_display, accuracies,
            metrics_by_snp, mafs, snp_ids
        )
        history['validation'].append(metrics_by_maf)

    end_time = time.time()
    print(
        f'Total elapsed time (seconds): {end_time - start_time}'
        f'Total training time (seconds): {end_time - time_before_training}'
    )

    if results_path:
        with open(f'{results_path}/aggregated-metrics-history.txt', 'w') as model_file:
            model_file.write(str(history))

    if args.plot:
        plot_metric(history, 'training', 'mcc', results_path=results_path)
        plot_metric(history, 'training', 'accuracy', results_path=results_path)
        plot_metric(history, 'training', 'precision', results_path=results_path)
        plot_metric(history, 'training', 'recall', results_path=results_path)
        plot_metric(history, 'training', 'f1-score', results_path=results_path)
        plot_metric(history, 'validation', 'mcc', results_path=results_path)
        plot_metric(history, 'validation', 'accuracy', results_path=results_path)
        plot_metric(history, 'validation', 'precision', results_path=results_path)
        plot_metric(history, 'validation', 'recall', results_path=results_path)
        plot_metric(history, 'validation', 'f1-score', results_path=results_path)
