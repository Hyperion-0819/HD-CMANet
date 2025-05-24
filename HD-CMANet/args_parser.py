import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='HDCMANet',
                        choices=[
                            # the proposed method
                            'HDCMANet'
                            # these  models are used for comparison experiments
                            'SSFCNN', 'ConSSFCNN',
                            'TFNet', 'ResTFNet',
                            'MSDCNN',
                            'SSRNET', 'MCT', 'MoGDCN', 'PSRT','MDC'
                        ])

    parser.add_argument('-root', type=str, default='./data')
    parser.add_argument('-dataset', type=str, default='Houston_HSI',
                        choices=['PaviaU', 'Botswana', 'KSC', 'Urban', 'Pavia', 'IndianP', 'Washington','MUUFL_HSI','salinas_corrected','Houston_HSI', 'Chikusei'])
    parser.add_argument('--scale_ratio', type=float, default=4)
    parser.add_argument('--n_bands', type=int, default=0)
    parser.add_argument('--n_select_bands', type=int, default=5)

    parser.add_argument('--model_path', type=str,
                        default='./checkpoints/dataset_arch.pkl',
                        help='path for trained encoder')
    parser.add_argument('--train_dir', type=str, default='./data/dataset/train',
                        help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='./data/dataset/val',
                        help='directory for resized images')

    # learning settingl
    parser.add_argument('--n_epochs', type=int, default=20000,
                        help='end epoch for training')
    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=128)

    args = parser.parse_args()
    return args
