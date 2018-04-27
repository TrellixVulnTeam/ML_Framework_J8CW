from services.learning_curves_service import LearningCurvesService
from classifiers.spongebob_character_classifier import SpongebobCharacterClassifier
from services.data_preprocessor_service import DataPreprocessorService as dps
from services.layer_initializer_service import LayerInitializerService
from models.data_model import DataModel
import matplotlib.pyplot as plt


data = dps.load_data()
data_model = DataModel(data, 7, [100, 100])
layers = LayerInitializerService.load_layers(7, 0.01)
classifier = SpongebobCharacterClassifier(data_model, 1000, layers)

learning_curves = LearningCurvesService.compute_learning_curves([classifier.data.x_train, classifier.data.y_train], [classifier.data.x_val, classifier.data.y_val], classifier, 5)

plt.plot(range(1, len(learning_curves['train_cost']) + 1), learning_curves['train_cost'])
plt.plot(range(1, len(learning_curves['val_cost']) + 1), learning_curves['val_cost'])
plt.show()
