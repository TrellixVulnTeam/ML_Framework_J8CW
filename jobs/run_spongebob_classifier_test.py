from tests.spongebob_character_classifier_test import SpongebobCharacterClassifierTest
import matplotlib.pyplot as plt


test = SpongebobCharacterClassifierTest(7, 'train')
classifier = test.run()

jhist = classifier.cost_history
plt.title = 'Cost History'
plt.plot(range(0, 1000), jhist)
plt.show()
