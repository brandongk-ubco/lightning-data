from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from tqdm import tqdm


def analyze(data: str):

    Dataset = DATAMODULE_REGISTRY[data]

    dataset = Dataset(num_workers=0, batch_size=1)

    dataloader = dataset.train_dataloader()
    dataiter = iter(dataloader)

    for sample in tqdm(dataiter):
        import pdb
        pdb.set_trace()
