import argparse

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--network-model',type=str,default='vgg16',help="select network model")
    parser.add_argument('--fine_tuning',type=str,default='all',
                        help="how to fine-tuning the network,if it is 'all',all variables will be changed in the train")
    parser.add_argument('--train_step',type=int,default=1000,help="The train step")
    parser.add_argument('--learning_rate',type=float,default=0.001,help="learning rate")
    return parser.parse_args()