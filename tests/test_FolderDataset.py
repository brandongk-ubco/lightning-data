import pathlib
from lightningdata.vision.classification import FolderDataset
import os


class TestFolderDataset:

    def test_no_files(self):
        folderpath = os.path.join(
            pathlib.Path(__file__).parent.absolute(), "data_dir", "folder")
        folder_dataset = FolderDataset(folderpath, extensions=[])
        assert len(folder_dataset) == 0

    def test_all_files(self):
        folderpath = os.path.join(
            pathlib.Path(__file__).parent.absolute(), "data_dir", "folder")
        folder_dataset = FolderDataset(folderpath)
        assert folder_dataset.classes
        assert len(folder_dataset) == 9
        assert len(folder_dataset.classes) == 2
        assert folder_dataset.classes[0] == 'a'
        assert folder_dataset.classes[1] == 'b'

        for i in range(9):
            folder_dataset.__getitem__(i)
