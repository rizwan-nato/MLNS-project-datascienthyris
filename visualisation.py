import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_results(idx):

    with open('results\\raw\\'+str(idx),'r') as f:
        hyperparamaters = f.read().split('_')
    BATCH_SIZE = int(hyperparamaters[0])
    LR = float(hyperparamaters[1])
    EPOCHS = int(hyperparamaters[2])
    hidden_size = int(hyperparamaters[3])
    dropout_p = float(hyperparamaters[4])
    n_layers = int(hyperparamaters[5])

    with open('results\\raw\\epoch_list_train_'+str(idx),'rb') as f:
        epoch_list_train = pickle.load(f)

    with open('results\\raw\\epoch_list_test_'+str(idx),'rb') as f:
        epoch_list_test = pickle.load(f)

    with open('results\\raw\\f1_score_list_train_'+str(idx),'rb') as f:
        f1_score_list_train = pickle.load(f)

    with open('results\\raw\\f1_score_list_test_'+str(idx),'rb') as f:
        f1_score_list_test = pickle.load(f)

    with open('results\\raw\\loss_list_train_'+str(idx),'rb') as f:
        loss_list_train = pickle.load(f)

    with open('results\\raw\\loss_list_test_'+str(idx),'rb') as f:
        loss_list_test = pickle.load(f) 

    return (epoch_list_train, 
            epoch_list_test, 
            f1_score_list_train, 
            f1_score_list_test, 
            loss_list_train, 
            loss_list_test,
            BATCH_SIZE,
            LR,
            EPOCHS,
            hidden_size,
            dropout_p,
            n_layers)

def plot_results(idx, save=False):
    epoch_list_train, epoch_list_test, f1_score_list_train, f1_score_list_test, loss_list_train, \
    loss_list_test, BATCH_SIZE, LR, EPOCHS, hidden_size, dropout_p, n_layers = load_results(idx)

    title = f'Batch size of {BATCH_SIZE}, learning rate of {LR} with {n_layers} hidden layers of size {hidden_size}'

    fig, ax1 = plt.subplots()
    fig.suptitle(title)
    ax2 = ax1.twinx() 

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('f1 score',color='red')
    ax2.set_ylabel('loss',color='blue')

    ax1.plot(epoch_list_train,f1_score_list_train,'--',label='f1 train',color='red')
    ax1.plot(epoch_list_test,f1_score_list_test,label='f1 test',color='red')
    ax2.plot(epoch_list_train,loss_list_train,'--',label='loss train',color='blue')
    ax2.plot(epoch_list_test,loss_list_test,label='loss test',color='blue')

    ax1.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.legend(bbox_to_anchor=(1.1, 1.05))
    ax2.legend(bbox_to_anchor=(1.37, 0.85))

    if save:
        plt.savefig('results\\plots\\'+str(idx)+'.png')

for i in range(17):
    plot_results(i,save=True)