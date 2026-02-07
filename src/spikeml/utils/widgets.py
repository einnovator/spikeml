from tqdm.notebook import tqdm

def _tqdm(data, total=None, progress=True):
    if progress:
        if total is not None:
           return tqdm(data, total=total)
        else:
           return tqdm(data)
    else:
        return data
