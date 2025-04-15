import kagglehub
import kagglehub.config
import shutil
import os

from path import Path

def download_dataset() -> None:
    ASSETS_FOLDER: str = Path(__file__).parent.parent / "assets" / "raw"

    try: assert ASSETS_FOLDER.exists()
    except AssertionError: print("Resources folder not found."); raise AssertionError

    kaggle_dataset_cache_folder: Path = Path(kagglehub.config.get_cache_folder()+"/datasets/ankkur13")
    
    if kaggle_dataset_cache_folder.exists(): 
        shutil.rmtree(kaggle_dataset_cache_folder)
    
    dataset_path: str = kagglehub.dataset_download(handle="ashishjangra27/doodle-dataset")

    if os.name == "nt":
        for file in Path(dataset_path).iterdir():
            shutil.move(str(file), str(RAW_DATA_FOLDER))
        shutil.rmtree(kaggle_dataset_cache_folder, ignore_errors=True)
    else:  # Linux/macOS
        assert not os.system(f"mv -v {dataset_path}/* {ASSETS_FOLDER}")
        assert not os.system(f"rm -rf {kaggle_dataset_cache_folder}")


if __name__ == "__main__": 
    download_dataset()
