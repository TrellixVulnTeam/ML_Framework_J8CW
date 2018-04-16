

class PredictionHelper:

    @staticmethod
    def predict(Z):
        return Z.argmax(axis=1)
