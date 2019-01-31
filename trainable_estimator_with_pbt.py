import datetime
import numpy as np
import ray
import tensorflow as tf
from ray.tune import Trainable, grid_search, run_experiments
from ray.tune.schedulers import PopulationBasedTraining
from tensorflow.python.data.experimental import CheckpointInputPipelineHook

import iris_data

tf.logging.set_verbosity(tf.logging.DEBUG)


class MyTrainableEstimator(Trainable):
    """
    Example how to combine tf.Estimator with ray.tune population based training (PBT)
    example is based on:
    TensorFlow doc examples:
    https://github.com/tensorflow/models/blob/master/samples/core/get_started/premade_estimator.py
    and estimator with raytune example by @sseveran:
    https://github.com/sseveran/ray-tensorflow-trainable/blob/master/estimator.py
    data loaded from script:
    https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py
    """

    def _setup(self, config):
        """
        Setup your tensorflow model
        :param config:
        :return:
        """

        # Hyperparameters for this trial can be accessed in dictionary self.config
        self.config = config

        # save all checkpoints independently to ray.tune checkpoint_path to avoid deleting by
        # tune.trainable.restore_from_object model dir based on timestamp
        if self.config['exp'] in self.logdir:
            self.model_dir = self.logdir.split(self.config['exp'])[0] + self.config['exp']
        else:
            raise IndexError(self.logdir + ' does not contain splitter ' + self.config['exp'] + 'check configuration '
                                                                                                'logdir path and exp '
                                                                                                'filed')
        self.model_dir_full = self.model_dir + '/' + datetime.datetime.now().strftime("%d_%b_%Y_%I_%M_%S_%f%p")

        # configuration
        self.training_steps = 250
        self.run_config = tf.estimator.RunConfig(
            save_summary_steps=100,
            save_checkpoints_secs=None,  # save checkpoint only before and after self.estimator.train()
            save_checkpoints_steps=self.training_steps,  # save both iterator checkpoint and model checkpoints after
            # same number of steps
            keep_checkpoint_max=None,  # avoid removing checkpoints
            keep_checkpoint_every_n_hours=None)

        # load data
        (train_x, train_y), (test_x, test_y) = iris_data.load_data()
        # Feature columns describe how to use the input.
        self.my_feature_columns = []
        for key in train_x.keys():
            self.my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        # estimator
        # Build 2 hidden layer DNN with 10, 10 units respectively.
        self.estimator = tf.estimator.DNNClassifier(
            feature_columns=self.my_feature_columns,
            # Two hidden layers of 10 nodes each.
            hidden_units=[10, 10],
            # The model must choose between 3 classes.
            n_classes=3,
            optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=self.config['lr'],
                l1_regularization_strength=self.config['l1'],

            ))

        # data for evaluation
        self.input_fn_eval = lambda: iris_data.eval_input_fn(test_x, test_y,
                                                             batch_size=20)
        # data for train
        self.input_fn_train = lambda: iris_data.train_input_fn(train_x, train_y,
                                                               batch_size=20)
        self.steps = 0

    def _train(self):
        # Run your training op for n iterations

        # possible to run evaluation in memory with:
        # evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(
        # self.estimator, self.input_fn_eval)

        # CheckpointInputPipelineHook save data iterator to resume loading data
        # from checkpoint. Otherwise iterator is initialized every time new estimator is created and iteration starts
        # from start point which might cause overfitting for big data models. CheckpointInputPipelineHook(...) use
        # self.run_config.save_checkpoints_secs or save_checkpoints_steps to save iterator. for more control over
        # saving read more: https://www.tensorflow.org/api_docs/python/tf/contrib/data/CheckpointInputPipelineHook
        self.datahook = CheckpointInputPipelineHook(self.estimator)
        # training
        self.estimator.train(input_fn=self.input_fn_train, steps=self.training_steps, hooks=[self.datahook])
        # evaluation
        metrics = self.estimator.evaluate(input_fn=self.input_fn_eval)
        self.steps = self.steps + self.training_steps
        return metrics

    def _stop(self):
        self.estimator = None

    def _save(self, checkpoint_dir):
        """
         This function will be called if a population member is good enough to be exploited
        :param checkpoint_dir:
        :return:
        """
        lastest_checkpoint = self.estimator.latest_checkpoint()
        # lastest_checkpoint = tf.contrib.training.wait_for_new_checkpoint(
        #    checkpoint_dir=self.model_dir_full,
        #    last_checkpoint=self.estimator.latest_checkpoint(),
        #    seconds_to_sleep=0.01,
        #    timeout=60
        # )
        #
        tf.logging.info('Saving checkpoint {} for tune'.format(lastest_checkpoint))
        f = open(checkpoint_dir + '/path.txt', 'w')
        f.write(lastest_checkpoint)
        f.flush()
        f.close()
        return checkpoint_dir + '/path.txt'

    def _restore(self, checkpoint_path):
        """
        Population members that perform very well will be
        exploited (restored) from their checkpoint
        :param checkpoint_path:
        :return:
        """
        f = open(checkpoint_path, 'r')
        path = f.readline().strip()
        tf.logging.info('Opening checkpoint {} for tune'.format(path))
        f.flush()
        f.close()
        tf.estimator.DNNClassifier(
            feature_columns=self.my_feature_columns,
            # Two hidden layers of 10 nodes each.
            hidden_units=[10, 10],
            # The model must choose between 3 classes.
            n_classes=3,
            optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=self.config['lr'],
                l1_regularization_strength=self.config['l1'],

            ),
            warm_start_from=tf.estimator.WarmStartSettings(ckpt_to_initialize_from=path)  # restore waights from the
            # checkpoint

        )


if __name__ == '__main__':

    train_spec = {
        "run": MyTrainableEstimator,
        "resources_per_trial": {
            "cpu": 1,
            "gpu": 1
        },
        "stop": {
            "accuracy": 1.0,  # value of the loss to stop, check with attribute
            "training_iteration": 30,  # how many times train is invoked
        },
        "config": {
            "lr": grid_search([10 ** -2, 10 ** -5]),
            "l1": grid_search([10 ** -4, 10 ** -6]),
            'exp': 'estimator',  # the name of directory where training results are saved

        },
        "num_samples": 10,
        'local_dir': '~/SR/estimator/',
        'checkpoint_at_end': True

    }

    ray.init()

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="accuracy",
        perturbation_interval=5,
        hyperparam_mutations={
            "lr": lambda: np.random.uniform(0, 1),
            "l1": lambda: np.random.uniform(0, 1),
        })

    run_experiments({"pbt_estimator": train_spec}, scheduler=pbt)
