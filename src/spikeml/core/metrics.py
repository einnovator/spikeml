import numpy as np

from spikeml.core.runner import InferenceRunner
from spikeml.datasets.dataset import SimpleDataset, DataLoader
from spikeml.datasets.encoder import one_hot

def eval(model, xx, yy, nclasses=None):
    if nclasses==None:
        nclasses = np.unique(yy).shape[0]
    yy_ = one_hot(yy, nclasses)
    dataset = SimpleDataset(xx)
    loader = DataLoader(dataset, batch_size=-1)
    runner = InferenceRunner(model, loader, log_step=-1)
    yy_prob = runner.run()
    yy_pred = np.argmax(yy_prob, -1)
    accuracy = np.sum(yy==yy_pred)/float(yy.shape[0])
    
    # Confusion matrix
    cm = np.zeros((nclasses, nclasses), dtype=int)

    for y_true, y_pred in zip(yy, yy_pred):
        cm[y_true, y_pred] += 1

    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm
    }
    
    return yy_prob, yy_pred, metrics
    
    

def print_pred(xx, yy, yy_pred, yy_prob, labels=None, nclasses=None, details=False):
    if labels is not None:
        if nclasses is None:
            nclasses = len(labels)
        yy_ = one_hot(yy, nclasses) if details else None
        for i in range(xx.shape[0]):
            print(f'{i}: {labels[yy[i]]} ({yy[i]}) -> {labels[yy_pred[i]]} ({yy_pred[i]})',  f'{yy_[i]} -> {yy_prob[i]}' if details else '')
    else:        
        yy_ = one_hot(yy, nclasses) if details else None
        for i in range(xx.shape[0]):
            print(f'{i}: {yy[i]} ->{yy_pred[i]}',  f'{yy_[i]} -> {yy_prob[i]}' if details else '')
