import json
import dataset
import model
import torch
import engine

def run():
    # Load Dataset
    train_dataset = dataset.autoencoderDataset("bibtex_train.embeddings")
    test_dataset  = dataset.autoencoderDataset("bibtex_test.embeddings")

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle=True, num_workers =  4)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset, batch_size = 1, num_workers = 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    autoencoder_model = model.autoEncoder(100) # 100 is the input dimension
    autoencoder_model.to(device)

    # Creating the optimizer
    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=1e-3)

    for epoch in range(0,10):
        training_loss = engine.train(train_dataloader, autoencoder_model, optimizer, device)
        print("Epoch: {} Loss: {}".format(epoch+1, training_loss))

    # Model evaluation
    engine.eval(test_dataloader, autoencoder_model, device)

    # Generating the embeddings now
    train_set_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = False)
    test_set_dataloader  = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle = False)

    embed_list = engine.generate_low_dimensional_embeddings(train_set_dataloader, autoencoder_model, device)
    embed_list.extend(engine.generate_low_dimensional_embeddings(test_set_dataloader, autoencoder_model, device))

    with open("bibtex_low_dimension.embeddings", mode='w+') as file:
        for each_elem in embed_list:
            line_to_write = " ".join(str(v) for v in each_elem[0])+'\n'
            file.write(line_to_write)

if __name__ == "__main__":
    run()