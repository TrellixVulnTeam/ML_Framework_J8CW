from tests.spongebob_character_classifier_test import SpongebobCharacterClassifierTest


test = SpongebobCharacterClassifierTest(7, 1000)
classifier = test.run()
train_f1 = classifier.compute_f1_score('train')
cv_f1 = classifier.compute_f1_score('cv')
test_f1 = classifier.compute_f1_score('test')

print('Train F1 Score: ' + str(train_f1))
print('CV F1 Score: ' + str(cv_f1))
print('Test F1 Score: ' + str(test_f1))
