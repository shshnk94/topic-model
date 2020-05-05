import argparse
import os 
import sys

parser = argparse.ArgumentParser(description='Topic Model Driver')

parser.add_argument('--model', type=str, help='name of the model')
parser.add_argument('--mode', type=str, help='mode of the call - train/eval/test')
parser.add_argument('--epochs', type=str, help='number of training epochs')
parser.add_argument('--topics', type=str, help='number of topics to be trained on')
parser.add_argument('--lr', type=str, help='Learning rate')
parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--ckpt_path', type=str, help='checkpoint path (if any)')

args = parser.parse_args()

if args.model == 'etm':

    if args.dataset == '20ng':
        data_path = 'data/etm/20ng'
    else:
        data_path = 'ETM\/data\/nips'

    train_embeddings = '1'

    if args.mode == 'train':
        os.system('python ETM/main.py --mode ' + args.mode + 
                  ' --dataset ' + args.dataset +
                  ' --data_path ' + data_path +
                  ' --save_path ' + 'results/etm/20ng' +
                  ' --num_topics ' + args.topics +
                  ' --train_embeddings ' + train_embeddings +
                  ' --epochs ' + args.epochs +
                  ' --lr ' + args.lr)
    else:
        os.system('python ETM/main.py --mode ' + args.mode + 
                  ' --dataset ' + args.dataset +
                  ' --data_path ' + data_path +
                  ' --save_path ' + 'ETM/results' +
                  ' --num_topics ' + args.topics +
                  ' --train_embeddings ' + train_embeddings +
                  ' --lr ' + args.lr +
                  ' --tc ' + '1' + 
                  ' --td ' + '1' +
                  ' --load_from ' + args.ckpt_path)

if args.model == 'nvdm':

    if args.dataset == '20ng':
        data_path = 'nvdm\/data\/20news/'
        vocab_size = '5430'
    else:
        data_path = 'nvdm\/data\/nips'
        vocab_size = '20881'
    
    save_path = 'nvdm/results'

    if args.mode == 'train':
        os.system('python nvdm/nvdm.py --data_dir ' + data_path +
                  ' --n_topic ' + args.topics +
                  ' --epochs ' + args.epochs +
                  ' --vocab_size ' + vocab_size +
                  ' --learning_rate ' + args.lr + 
                  ' --save_path ' + save_path)
    else:
        test = 'True'
        os.system('python nvdm/nvdm.py --data_dir ' + data_path +
                  ' --test ' + test +
                  ' --n_topic ' + args.topics +
                  ' --epochs ' + args.epochs +
                  ' --vocab_size ' + vocab_size +
                  ' --learning_rate ' + args.lr +
                  ' --save_path ' + save_path)

if args.model == 'prodlda':
    
    if args.dataset == '20ng':
        data_path = 'autoencoding_vi_for_topic_models\/data\/20news_clean'
    else:
        data_path = 'autoencoding_vi_for_topic_models\/data\/nips'

    save_path = 'autoencoding_vi_for_topic_models\/results'

    os.system('CUDA_VISIBLE_DEVICES=0 python autoencoding_vi_for_topic_models/run.py -m prodlda' +
              ' -f 100' + 
              ' -s 100' +
              ' -t ' + args.topics +
              ' -b 200' + 
              ' -r 0.002' +
              ' -e ' + args.epochs +
              ' --data_path ' + data_path + 
              ' --save_path ' + save_path)
