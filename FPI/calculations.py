import os
import sys
import numpy as np
from PIL import Image


def LoadImageFolder(addr: str, resize: tuple = None) -> list:
    if not resize:
        return [
            np.array(Image.open(os.path.join(addr, image)).convert("L"))
            for image in os.listdir(addr)
        ]
    else:
        return [
            np.array(Image.open(os.path.join(addr, image)).resize(resize).convert("L"))
            for image in os.listdir(addr)
        ]


def LR_data(addr: str) -> list:
    images = LoadImageFolder(addr)
    ans = [[] for _ in range(len(images))]

    for row in range(images[0].shape[0]):
        [ans[i].append([]) for i in range(len(ans))]
        for col in range(images[0].shape[1]):
            NDVI = np.array([images[i][row][col] for i in range(len(images))])
            NDVI_max = max(NDVI)
            NDVI_min = min(NDVI)

            LR_max = 30 + 30 * (NDVI_max - 125) / (255 - 125)
            if NDVI_max != NDVI_min:
                RG = (NDVI - NDVI_min) / (NDVI_max - NDVI_min) * 100
            else:
                RG = NDVI * 0
            LR = RG * LR_max / 100
            for i in range(len(ans)):
                ans[i][-1].append(LR[i])
    return ans


def MR_data(addr: str) -> list:
    images = LoadImageFolder(addr, resize=(810, 451))
    ans = [[] for _ in range(len(images))]

    for row in range(images[0].shape[0]):
        [ans[i].append([]) for i in range(len(ans))]
        for col in range(images[0].shape[1]):
            H100 = np.array([images[i][row][col] for i in range(len(images))])
            H100_max = max(H100)
            H100_min = min(H100)
            if H100_max != H100_min:
                MR = (H100 - H100_min) / (H100_max - H100_min)
            else:
                MR = H100 * 0
            for i in range(len(ans)):
                ans[i][-1].append(MR[i])
    return ans


def FPI_calc(lr_addr: str, mr_addr: str) -> list:
    proc_schedule = [0, 0, 1, 1, 2, 2, 3]
    lrs = LR_data(lr_addr)
    mrs = MR_data(mr_addr)
    ans = []
    for image_id in range(len(lrs)):
        tmp = (60 - np.asarray(lrs[image_id], dtype=float)) * (
            1 - np.asarray(mrs[proc_schedule[image_id]], dtype=float)
        )
        tmp = tmp.tolist()
        ans.append(tmp)
    return ans


def arr_to_imgs(
    images: list, text: str, scale: int = 1.0, inv: bool = False, bias: int = 0
) -> None:
    for i, im in enumerate(images):
        im = np.asarray(im)
        im = np.uint8((im * scale) + bias)
        print(np.max(im), np.min(im))
        if inv:
            im = 255 - im
            print("inversed")
            print(np.max(im), np.min(im))
        Image.fromarray(im, "L").save(text + "-" + str(i) + ".png")


def make_gif(
    text: str, count: int, extension: str = ".png", duration: int = 300, loop: int = 0
) -> None:
    import imageio

    images = [imageio.imread(text + "-" + str(i) + extension) for i in range(count)]
    imageio.mimwrite(text + ".gif", images, fps=1)


if __name__ == "__main__":
    try:
        section = int(sys.argv[1])
        gif = int(sys.argv[2])
    except:
        raise ValueError("\n\nSelect one option and pass it as i/p parameter to the file for example:\n$ python calculations.py 0 0\nfirst parameter is for the selection of factors:\n0-FPI, 1-LR, 2-LR_inv, 3-MR, 4-MR_inv\nThe second parameter is for giving permission to make gif visualization or not (0/1)")
    if section == 0:
        images = FPI_calc("NDVI", "TRMM")
        arr_to_imgs(images, "FPI", scale=255 / 60)
        if gif:
            make_gif("FPI", 7)
        final = np.asarray(images) * 100 / 60
        np.save("FPI.npy", final)
    elif section == 1:
        images = LR_data("NDVI")
        arr_to_imgs(images, "LR", 2.55 * 100 / 60)
        if gif:
            make_gif("LR", 7)
    elif section == 2:
        images = LR_data("NDVI")
        arr_to_imgs(images, "LR_inv", scale=255 / 60, inv=True)
        if gif:
            make_gif("LR_inv", 7)
    elif section == 3:
        images = MR_data("TRMM")
        arr_to_imgs(images, "MR", 255)
        if gif:
            make_gif("MR", 4)
    elif section == 4:
        images = MR_data("TRMM")
        arr_to_imgs(images, "MR_inv", 150, inv=True, bias=105)
        if gif:
            make_gif("MR_inv", 4)

    print("Done")
