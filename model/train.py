from model.architecture import *
from config import *
from data_loader import *
from tqdm import tqdm

def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset, epochs):
### EBAUCHE A CHANGER QUANT YAURA LES DONNEES
    f1_score_list = []
    epoch_list = []
    for epoch in range(epochs):
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
        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))

        if epoch % 5 == 0:
            scores = []
            for batch, test_data in enumerate(test_dataset):
                subgraph, features, labels = test_data
                subgraph = subgraph.clone().to(device)
                features = features.clone().detach().to(device)
                labels = labels.clone().detach().to(device)
                score, _ = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
                scores.append(score)
                f1_score_list.append(score)
                epoch_list.append(epoch)
            print("F1-Score: {:.4f} ".format(np.array(scores).mean()))

    return epoch_list, f1_score_list

def evaluate(features, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
        output = model(features.float())
        loss_data = loss_fcn(output, labels.float())
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
    train_dataset, test_dataset = EEGDataset(mode="train"), EEGDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    n_features, n_classes = train_dataset[0][1].shape[1], 8
    model = model_class(g=train_dataset[0][0], input_size = n_features, output_size = 16, **model_args).to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    e,f = train(model, loss_fcn, device, optimizer, train_dataloader, test_dataloader, epochs)

    torch.save(model.state_dict(), MODEL_STATE_FILE)

    return e,f,model

if __name__ == '__main__':
    if MODEL_TYPE == 'CONV':
        model_class = BasicGraphModel
    e,f,model = train_pipeline(model_class, BATCH_SIZE, LR, EPOCHS, MODEL_ARGS)