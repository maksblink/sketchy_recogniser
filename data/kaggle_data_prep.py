import PIL
import PIL.Image
import PIL.ImageDraw
import pandas as pd

import tqdm
import json

from multiprocessing.pool import ThreadPool


# https://www.kaggle.com/code/mariasaif/building-and-training-a-cnn-for-doodle-classificat

from path import Path

ASSETS_FOLDER: str = Path(__file__).parent.parent / "assets" / "kaggle"

object_list: list[str] = ["house plant", "guitar", "basketball", "sword", "door", "key", "lantern", "chair", "pencil"]


def filter_data() -> None:
    master_file: Path = ASSETS_FOLDER / "master_doodle_dataframe.csv"
    df: pd.DataFrame = pd.read_csv(master_file, index_col=0)

    drop_cols: list[str] = ["countrycode", "key_id", "recognized", "image_path"]
    for col_name in drop_cols:
        if col_name in df.columns:
            df = df.drop(labels=col_name, axis=1)

    df = df[df["word"].isin(object_list)]
    df = df.reset_index(drop=True)

    df.to_csv(master_file)


def draw_strokes(image_draw: PIL.ImageDraw.ImageDraw, strokes: str) -> None:
    for stroke in json.loads(strokes):
        for i in range(0, len(stroke[0]) - 1, 1):
            x1, y1 = stroke[0][i], stroke[1][i]
            x2, y2 = stroke[0][i + 1], stroke[1][i + 1]
            image_draw.line(((x1, y1),(x2, y2)), fill=255, width=2)
    return


def create_images(): 
    master_file: Path = ASSETS_FOLDER / "master_doodle_dataframe.csv"
    df: pd.DataFrame = pd.read_csv(master_file)

    CLASSES_FOLDER: Path = ASSETS_FOLDER / "classes"
    CLASSES_FOLDER.mkdir()

    # Future work: add threadpool
    image_number: int = 0
    for image_class, strokes in zip(df["word"], df["drawing"]):
        image_class = str(image_class).translate({ord(" "): ord("_")})

        img: PIL.Image.Image = PIL.Image.new(size=(255, 255), color="black", mode="L")
        image_draw: PIL.ImageDraw.ImageDraw = PIL.ImageDraw.Draw(img)

        draw_strokes(image_draw, strokes)

        class_dir: Path = CLASSES_FOLDER / image_class
        if not class_dir.exists(): class_dir.mkdir()

        img.save(CLASSES_FOLDER / image_class / f"{image_class}_{image_number}.png")
        image_number += 1
        

if __name__ == "__main__":
    # filter_data()
    create_images()
