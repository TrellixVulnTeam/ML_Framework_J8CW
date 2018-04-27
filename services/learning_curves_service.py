

class LearningCurvesService:

    @staticmethod
    def compute_learning_curves(train_data, cv_data, classifier, batch_size):
        initial_classifier = classifier  # preserves state of classifier before training
        x_train, y_train = train_data
        x_val, y_val = cv_data

        costs = {'train_cost': [], 'val_cost': []}
        for i in range(int(len(x_train) / batch_size)):
            end = (i + 1) * batch_size

            # train on train data
            x_train_batch = x_train[0:end]
            y_train_batch = y_train[0:end]
            print(x_train_batch.shape)

            classifier.train(x_train_batch, y_train_batch)
            train_cost = classifier.compute_cost(y_train_batch, classifier.y_pred)

            # get cost on cv data
            x_val_batch = x_val[0:end]
            y_val_batch = y_val[0:end]

            y_val_pred = classifier.forward_propogate(x_val_batch)
            cv_cost = classifier.compute_cost(y_val_batch, y_val_pred)

            print(train_cost)
            costs['train_cost'].append(train_cost)
            costs['val_cost'].append(cv_cost)

            classifier = initial_classifier

        return costs
