from model import LSTMSentimentTagger
from torch.utils.data import Dataset, DataLoader
from data_loading import get_datasets_and_vectorizer
from pytorch_lightning import Trainer
import torch

if __name__=='__main__':
    config = {
        'batch_size': 32,
        'hidden_dim': 128,
        'num_layers': 1,
        'lr': 0.001,
        'embedding_dim': 50,
        'epochs': 10,
        'pretrained_embedding': False,
        'optimizer': 'Adam',
    }

    BATCH_SIZE = config['batch_size']

    train_dataset, valdiation_dataset, test_dataset, vectorizer = get_datasets_and_vectorizer()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = custom_collate_fn)
    validation_loader = DataLoader(valdiation_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn = custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn = custom_collate_fn)


    model = LSTMSentimentTagger(word_vector = vectorizer.word_vector,
                        hidden_dim=config['hidden_dim'],
                        embedding_dim=config['embedding_dim'],
                        num_layers=config['num_layers'],
                        pretrained_embedding=config['pretrained_embedding'])

    AVAIL_GPUS = min(1, torch.cuda.device_count())
    print(AVAIL_GPUS)

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=config['epochs'],
    )
    trainer.fit(model, train_loader, validation_loader)
    trainer.test(model, test_loader)

    model.save('model.pt')
    