from model.architecture import *
from config import *
from data_loader import *
from tqdm import tqdm
import time
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataloader, epochs):
### EBAUCHE A CHANGER QUANT YAURA LES DONNEES
    f1_score_list_train = []
    epoch_list_train = []
    loss_list_train = []

    f1_score_list_test = []
    epoch_list_test = []  
    loss_list_test = []

    for epoch in range(epochs):
        t_i = time.time()
        print(f'Epoch {epoch}/{epochs}')
        model.train()
        losses = []
        for batch, data in tqdm(enumerate(train_dataloader)):
            subgraph, features, labels = data
            subgraph = subgraph.to(device)
            features = features.to(device)
            labels = labels.to(device)
            model.g = subgraph
            for layer in model.layers:
                layer.g = subgraph
            logits = model(features.float())
            loss = loss_fcn(logits, labels.type(torch.LongTensor))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses, dtype=np.float32).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))
        print(f"Time: {int(time.time()-t_i)} seconds")

        if True:
            scores_train = []
            losses_train = []
            for _, test_data in enumerate(train_dataloader):
                subgraph, features, labels = test_data
                subgraph = subgraph.clone().to(device)
                features = features.clone().detach().to(device)
                labels = labels.clone().detach().to(device)
                f1_score, loss = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
                scores_train.append(f1_score)
                losses_train.append(loss)
            epoch_list_train.append(epoch)
            f1_score_list_train.append(np.array(scores_train).mean())
            loss_list_train.append(np.array(losses_train).mean())
            print("F1-Score on train: {:.4f} ".format(np.array(scores_train).mean()))

            scores_test = []
            losses_test = []
            for _, test_data in enumerate(test_dataloader):
                subgraph, features, labels = test_data
                subgraph = subgraph.clone().to(device)
                features = features.clone().detach().to(device)
                labels = labels.clone().detach().to(device)
                f1_score, loss = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
                scores_test.append(f1_score)
                losses_test.append(loss)
            epoch_list_test.append(epoch)
            f1_score_list_test.append(np.array(scores_test).mean())
            loss_list_test.append(np.array(losses_test).mean())
            print("F1-Score on test: {:.4f} ".format(np.array(scores_test).mean()))

    return epoch_list_train, f1_score_list_train, loss_list_train, epoch_list_test, f1_score_list_test, loss_list_test

def evaluate(features, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
        output = model(features.float())
        loss_data = loss_fcn(output, labels.type(torch.LongTensor))
        predict = np.argmax(output.data.cpu().numpy(), axis=1)
        score = f1_score(labels.data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()

def train_pipeline(
    model_class,
    batch_size,
    lr,
    epochs,
    model_args
    ):
    device = torch.device("cpu" if GPU < 0 else "cuda:" + str(GPU))
    train_dataset, test_dataset = EEGDataset(mode="train"), EEGDataset(mode="dev")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    n_features, n_classes = train_dataset[0][1].shape[1], 8
    model = model_class(g=train_dataset[0][0], input_size = n_features, output_size = 64, **model_args).to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    epoch_list_train, f1_score_list_train, loss_list_train, epoch_list_test, f1_score_list_test, loss_list_test = train(model, loss_fcn, device, optimizer, train_dataloader, test_dataloader, epochs)

    torch.save(model.state_dict(), MODEL_STATE_FILE)

    return epoch_list_train, f1_score_list_train, loss_list_train, epoch_list_test, f1_score_list_test, loss_list_test, model

if __name__ == '__main__':
    if MODEL_TYPE == 'CONV':
        model_class = BasicGraphModel

    k = 18
    EPOCHS = 5

    # BATCH_SIZE = 64
    # LR = 0.0001
    
    # MODEL_ARGS['hidden_size'] = 32
    # MODEL_ARGS['dropout_p'] = 0.5
    # MODEL_ARGS['n_layers'] = 2

    for BATCH_SIZE in [16, 32, 64]:
        # for LR in [0.000001, 0.00001, 0.0001, 0.001]:
        for LR in [0.0001, 0.001]:
            for hidden_size in [8, 16, 32]:
                for n_layers in [1, 2, 3]:

                    MODEL_ARGS = {}
                    MODEL_ARGS['hidden_size'] = hidden_size
                    MODEL_ARGS['dropout_p'] = 0.5
                    MODEL_ARGS['n_layers'] = n_layers

                    epoch_list_train, f1_score_list_train, loss_list_train, epoch_list_test, f1_score_list_test, loss_list_test, model = train_pipeline(model_class, BATCH_SIZE, LR, EPOCHS, MODEL_ARGS)

                    idx = str(k)
                    name_txt = str(BATCH_SIZE)+'_'+str(LR)+'_'+str(EPOCHS)+'_'+str(MODEL_ARGS['hidden_size'])+'_'+str(MODEL_ARGS['dropout_p'])+'_'+str(MODEL_ARGS['n_layers'])
                    # model.save('model\models\\'+hyperparameters)

                    with open('results\\raw\\epoch_list_train_'+idx, 'wb') as f:
                        pickle.dump(epoch_list_train, f)

                    with open('results\\raw\\f1_score_list_train_'+idx, 'wb') as f:
                        pickle.dump(f1_score_list_train, f)

                    with open('results\\raw\\loss_list_train_'+idx, 'wb') as f:
                        pickle.dump(loss_list_train, f)

                    with open('results\\raw\\epoch_list_test_'+idx, 'wb') as f:
                        pickle.dump(epoch_list_test, f)

                    with open('results\\raw\\f1_score_list_test_'+idx, 'wb') as f:
                        pickle.dump(f1_score_list_test, f)

                    with open('results\\raw\\loss_list_test_'+idx, 'wb') as f:
                        pickle.dump(loss_list_test, f)

                    with open('results\\raw\\'+idx, 'w') as f:
                        f.write(name_txt)

                    k += 1

