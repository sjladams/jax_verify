"""Run verification for feedforward ReLU networks."""

import os
from typing import Any, Callable, Mapping
import numpy as np
from absl import app
from absl import logging
from jax_verify.extensions.functional_lagrangian import bounding
from jax_verify.extensions.functional_lagrangian import data
from jax_verify.extensions.functional_lagrangian import model
from jax_verify.extensions.functional_lagrangian import verify_utils

import pickle

from configs.config_adv_stochastic_model import get_config


def pickle_dump(object, tag):
    pickle_out = open('{}.pickle'.format(tag), 'wb')
    pickle.dump(object, pickle_out)
    pickle_out.close()


def make_logger(log_message: str) -> Callable[[int, Mapping[str, Any]], None]:
  """Creates a logger.

  Args:
    log_message: description message for the logs.

  Returns:
    Function that accepts a step counter and measurements, and logs them.
  """

  def log_fn(step, measures):
    msg = f'[{log_message}] step={step}'
    for k, v in measures.items():
      msg += f', {k}={v}'
    logging.info(msg)

  return log_fn


def main(unused_argv):
    N = 50
    epsilon = 0.025

    config = get_config(model_name='mnist_mlp_1_512')

    logging.info('Config: \n %s', config)

    config.problem.epsilon_unprocessed = epsilon # radius before preprocessing
    store = {'bounds': np.zeros((N,10)), 'true_labels': np.zeros(N)}
    for dataset_idx in range(N):
        config.problem.dataset_idx = dataset_idx  # which example from dataset to verify?

        for label_idx in range(10):
            config.problem.target_label_idx = label_idx  # which class to target?

            data_spec = data.make_data_spec(config.problem, config.assets_dir)
            spec_type = {e.value: e for e in verify_utils.SpecType}[config.spec_type]

            if data_spec.true_label == data_spec.target_label:
                continue
            else:
                params = model.load_model(
                  root_dir=config.assets_dir,
                  model_name=config.problem.model_name,
                  num_std_for_bound=config.problem.get('num_std_for_bound'),
                )

                params_elided, bounds, bp_bound, bp_time = (
                  bounding.make_elided_params_and_bounds(config, data_spec, spec_type,
                                                         params))
                ub = bounds[2].ub
                lb = bounds[2].lb
                store['bounds'][dataset_idx, label_idx] = bp_bound

        store['true_labels'] = data_spec.true_label

    store['robust'] = np.all(store['bounds'] <= 0, axis=1)

    pickle_dump(store, 'summary_mnist_deepmind_ib_logits_{}_{}'.format(epsilon, N))

    print('to process')
    x = 0


if __name__ == '__main__':
    app.run(main)
