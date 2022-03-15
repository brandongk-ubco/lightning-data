from lightningdata import Datasets
from argh import arg


@arg('dataset', choices=Datasets.choices())
def initialize(dataset: Datasets):

    Dataset = Datasets.get(dataset)
    dataset = Dataset()
    dataset.initialize()
