import PIL.Image
import pandas as pd
import numpy as np
import cv2
import tqdm
import PIL

# https://www.kaggle.com/code/mariasaif/building-and-training-a-cnn-for-doodle-classificat

from path import Path

ASSETS_FOLDER: str = Path(__file__).parent.parent / "assets" / "kaggle"

object_list: list[str] = ["house plant", "guitar", "basketball", "sword", "door", "key", "lantern", "chair", "pencil"]


def strokes_to_image(strokes, size=(255, 255)):
    img = np.zeros(size, dtype=np.uint8)
    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            x1, y1 = stroke[0][i], stroke[1][i]
            x2, y2 = stroke[0][i + 1], stroke[1][i + 1]
            img = cv2.line(img, (x1, y1), (x2, y2), 255, 2)
    return img


# def convert_drawings_to_images(dataframe):
#     images = []
#     for drawing in tqdm(dataframe['drawing'], desc='Converting drawings to images'):
#         strokes = ast.literal_eval(drawing)
#         img = strokes_to_image(strokes, size=(64, 64))
#         images.append(img)
#     return np.array(images)


def create_images(): 
    master_file: Path = ASSETS_FOLDER / "master_doodle_dataframe.csv"

    df: pd.DataFrame = pd.read_csv(master_file)
    
    df = df.drop(labels=["countrycode", "key_id", "recognized", "image_path"], axis=1)
    print(df["drawing"][0])

    image = PIL.Image.fromarray(strokes_to_image(df["drawing"][0]))
    image.save("test.png")
    

if __name__ == "__main__":
    create_images()

