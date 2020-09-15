import argparse

parser = argparse.ArgumentParser(description='Learning Deep Priors for Image Dehazing')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/mnt/lustre/liuyang2/2019-dehaze/dehaze-data_v4',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='dehazedata',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='dehaze_test',
                    help='test dataset name')
parser.add_argument('--ext', type=str, default='img',
                    help='dataset file extension')
parser.add_argument('--scale', default='1',
                    help='super resolution scale which is of no use for dehazing')
parser.add_argument('--patch_size', type=int, default=512,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1.0,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')

# Model specifications
parser.add_argument('--model', default='dehaze_net',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=5,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=8,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--weight_A_prior', type=float, default=0.01,
                    help='weight of A_prior')
parser.add_argument('--weight_t_prior', type=float, default=0.01,
                    help='weight of t_prior')
parser.add_argument('--weight_J_prior', type=float, default=0.01,
                    help='weight of J_prior')
parser.add_argument('--t_clamp', type=float, default=0.1,
                    help='clamp the value of t')
parser.add_argument('--weight_A_prior_phase1', type=float, default=0.01,
                    help='weight of A_prior')
parser.add_argument('--weight_t_prior_phase1', type=float, default=0.01,
                    help='weight of t_prior')
parser.add_argument('--weight_J_prior_phase1', type=float, default=0.01,
                    help='weight of J_prior')
parser.add_argument('--weight_A_prior_phase2', type=float, default=0.001,
                    help='weight of A_prior')
parser.add_argument('--weight_t_prior_phase2', type=float, default=0.001,
                    help='weight of t_prior')
parser.add_argument('--weight_J_prior_phase2', type=float, default=0.001,
                    help='weight of J_prior')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=20,
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=10,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='dehaze_model',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=115,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')

args = parser.parse_args()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

