import Algorithmia
import cv2
import numpy as np



client = Algorithmia.client("simgZvKSdeCqKEejmgnRpSxx2Zp1")

input = {
    "image": "C:\\Users\\rahul\\Desktop\\rahul.npy",
    "numResults": 7
}

algo = client.algo('deeplearning/EmotionRecognitionCNNMBP/0.1.2')
algo.set_options(timeout=300)
result = algo.pipe(input).result

print(result)