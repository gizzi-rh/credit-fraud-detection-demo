import keras
import onnx


def saveModel():
    model = onnx.load("fraud.onnx")
    keras.models.save_model(model, filepath="fraud/1")


if __name__ == "__main__":
    saveModel()