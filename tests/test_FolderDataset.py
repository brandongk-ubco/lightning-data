import pathlib
from lightningdata.vision.classification import FolderDataset
import os


class TestFolderDataset:

    def test_something(self):
        folderpath = os.path.join(
            pathlib.Path(__file__).parent.absolute(), "data_dir", "folder")
        folder_dataset = FolderDataset(folderpath, extensions=[])
        assert len(folder_dataset) == 0
