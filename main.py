import os, argparse, torch, random, sys
from Trainer import Trainer
from Load_model import Load_model, Load_data
from util import check_folder
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pandas as pd
import pdb

# fix random
SEED = 999
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='V1') # self-defined model version
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=400) # epoch
    parser.add_argument('--batch_size', type=int, default=8) # batch
    parser.add_argument('--lr', type=float, default=0.00005) # learning rate
    parser.add_argument('--loss', type=str, default='l1') # objective function
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--model', type=str, default='BLSTM') # model
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--target', type=str, default='MAP') #'MAP' or 'IRM'
    parser.add_argument('--task', type=str, default='VCTK') # dataset
    parser.add_argument('--resume' , action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--save_results', type=str, default='False') # whether save the processed speech
    parser.add_argument('--re_epochs', type=int, default=300)
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()
    return args

def get_path(args):
    
    checkpoint_path = f'checkpoint\\'\
    f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
    f'_{args.loss}_batch{args.batch_size}_lr{args.lr}.pth.tar'
    
    model_path = f'save_model\\'\
    f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
    f'_{args.loss}_batch{args.batch_size}_lr{args.lr}.pth.tar'
    
    score_path = {
    'PESQ':f'sourc\\PESQ\\'\
    f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
    f'_{args.loss}_batch{args.batch_size}_lr{args.lr}.csv',       
    'STOI':f'sourc\\STOI\\'\
    f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
    f'_{args.loss}_batch{args.batch_size}_lr{args.lr}.csv'
    }
    
    return checkpoint_path,model_path,score_path

if __name__ == '__main__':
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    print('random seed =', SEED)
        
    # get parameter
    args = get_args() # 從command line中獲取所需的參數
    
    print('model name =', args.model)
    print('target mode =', args.target)
    print('version =', args.version)
    print('Lr = ', args.lr)
    
    # data path
    Train_path = {
    'noisy':'C:\\D\\77_Program\\Intern\\Intern_SE\\VCTK_28spk\\noisy_trainset_wav', # input
    'clean':'C:\\D\\77_Program\\Intern\\Intern_SE\\VCTK_28spk\\clean_trainset_wav' # target
    } 

    Test_path = {
    'noisy':'C:\\D\\77_Program\\Intern\\Intern_SE\\VCTK_28spk\\noisy_testset_wav',
    'clean':'C:\\D\\77_Program\\Intern\\Intern_SE\\VCTK_28spk\\clean_testset_wav'
    }
        
    Output_path = {
    'audio':f'result\\'\
        f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
        f'_{args.loss}_batch{args.batch_size}_lr{args.lr}'
    }
    
    # declair path
    checkpoint_path,model_path,score_path = get_path(args)

    # tensorboard
    writer = SummaryWriter(f'logs\\'\
                           f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
                           f'_{args.loss}_batch{args.batch_size}_lr{args.lr}')
    
    # pdb.set_trace()
    exec (f"from models.{args.model.split('_')[0]} import {args.model} as model") # The type of model is decided by the command line
    model     = model() # model -- from the instruction above
    model, epoch, best_loss, optimizer, scheduler, criterion, device = Load_model(args,model,checkpoint_path, model_path)
    
    loader = Load_data(args, Train_path)
    if args.retrain:
        args.epochs = args.re_epochs 
        checkpoint_path, model_path, score_path = get_path(args)
        
    # pdb.set_trace()    
    Trainer = Trainer(model, args.version, args.epochs, epoch, best_loss, optimizer,scheduler, 
                      criterion, device, loader, Test_path, writer, model_path, score_path, args, Output_path, args.save_results, args.target)
    try:
        if args.mode == 'train':
            Trainer.train()
        Trainer.test()
        
    except KeyboardInterrupt: # 執行被中斷的話，儲存目前的進度
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
            }
        check_folder(checkpoint_path)
        torch.save(state_dict, checkpoint_path)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
