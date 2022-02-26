from model import LSTMSentimentTagger
from data_loading import get_datasets_and_vectorizer, PADDING_VALUE
import torchtext
import torch

if __name__=='__main__':

    review = 'This is just an over hyped, overrated movie that put me to sleep. During the hype before watching this I expected to see something great. I did not. The movie was so boring and so long. It was a long boring movie.'

    _, _, _, vectorizer = get_datasets_and_vectorizer()
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
    model = LSTMSentimentTagger(word_vector = vectorizer.word_vector,
                        hidden_dim=config['hidden_dim'],
                        embedding_dim=config['embedding_dim'],
                        num_layers=config['num_layers'],
                        pretrained_embedding=config['pretrained_embedding'])
    
    model.load('model.pt')
    model.eval()
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english', language='en')
    
    seq = review
    length = len(tokenizer(seq))
    seq = [vectorizer.vectorize(tokenizer(seq))]
    seq = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=PADDING_VALUE)
    with torch.no_grad():
        print(model(seq, [length]))
    