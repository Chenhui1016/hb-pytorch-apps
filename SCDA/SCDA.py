import argparse
import time
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
        use_bias = True # TODO: check if setting this to False helps with unbalanced classes
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


#-------------------------------------------------------------
# Helpers
#-------------------------------------------------------------

# convert dataframe to torch tensor
def dataframe_to_tensor(df):
    return torch.from_numpy(df.values)

# compute loss to show during training
# use reduction=none to keep individual feat losses separate and aggregate by MAF later
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
    batch_acc = true_positives.sum(dim=1) / y.size(1) # for each instance in batch, correct SNPs / total SNPs
    # TODO: get accuracies by SNP to aggregate by maf
    cpu_batch_acc = batch_acc.cpu().numpy()
    return cpu_batch_acc

# print average loss for each MAF category
def print_loss_by_maf(subset_name, epoch, losses, accuracies, mafs, snps):
    VERY_LOW = '[0-0.5%)'
    LOW = '[0.5-5%)'
    HIGH = '[5-50%)'
    # initialize accumulator for each MAF category
    loss_by_maf = {
        bucket: {
            # TODO: add accuracies / MCCs
            'loss': 0.0,
            'instances': 0
        } for bucket in (VERY_LOW, LOW, HIGH)
    }

    avg_loss_by_feat = np.mean(losses, axis=0)

    for index, snp_id in enumerate(snps):
        maf = mafs[snp_id][0]
        if maf < 0.005:
            bucket = VERY_LOW
        elif maf < 0.05:
            bucket = LOW
        else:
            bucket = HIGH
        loss_by_maf[bucket]['loss'] += avg_loss_by_feat[index]
        loss_by_maf[bucket]['instances'] += 1
    print(
        f'Epoch {epoch}; {subset_name} set\n'
        f'Loss={np.mean(avg_loss_by_feat):.6f}\n'
        f'Average accuracy={np.mean(accuracies):.6f}\n'
        f'By MAF: {loss_by_maf}\n'
    )


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


#-------------------------------------------------------------
# Main
#-------------------------------------------------------------

if __name__ == '__main__':
    start_time = time.time()

    args = parser.parse_args()

    input_path = args.input
    whole_dataframe = pd.read_csv(input_path, sep='\t', index_col=0)
    mafs = pd.read_csv(args.mafs_path)

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
    validation_tensor = dataframe_to_tensor(valid_df)
    training_one_hot = nn.functional.one_hot(training_tensor).float()
    validation_one_hot = nn.functional.one_hot(validation_tensor).float()
    print(f'training set shape: {str(training_one_hot.size())}')
    print(f'validation set shape: {str(validation_one_hot.size())}')

    # init SCDA model
    model = SCDA(
        training_one_hot.size(-1), args.channels, args.dropout_amount,
        args.filter_size
    )
    if args.hammerblade:
        device = torch.device('hammerblade')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print()
    print(model)
    print()

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

    snp_ids = whole_dataframe.columns.values

    time_before_training = time.time()
    for epoch in range(args.max_epochs):
        model.train()
        losses_for_display = []
        accuracies = []
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
            accuracies = np.concatenate((accuracies, batch_accuracy(z, y)))

            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

        print_loss_by_maf(
            'training', epoch, losses_for_display, accuracies, mafs, snp_ids
        )

        # validation
        model.eval()
        losses_for_display = []
        accuracies = []
        for batch_idx, (x, y) in tqdm(
            enumerate(validation_generator), total=len(validation_generator)
        ):
            x, y = x.contiguous(), y.contiguous()
            x, y = x.to(device), y.to(device)

            z = model(x)

            losses_for_display.append(batch_loss_for_display(z, y))
            accuracies = np.concatenate((accuracies, batch_accuracy(z, y)))

        print_loss_by_maf(
            'validation', epoch, losses_for_display, accuracies, mafs, snp_ids
        )

    end_time = time.time()
    print(
        f'Total elapsed time (seconds): {end_time - start_time}'
        f'Total training time (seconds): {end_time - time_before_training}'
    )
