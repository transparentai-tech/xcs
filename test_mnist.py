import logging
import math
import time

import xcs
from xcs import scenarios
from xcs.conditions.base import IntervalCondition


def test_mnist(epochs: int = 10):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class_count = 10
    max_rules = 1000
    max_rules_per_class = max_rules / class_count
    input_size = 28 ** 2

    # Upper & lower bounds for each pixel. Ignoring effect of others, each bound cuts space in half => 1 bit of info.
    max_bits_per_rule = 2 * input_size

    bits_per_class = math.log2(10)
    target_specified_bits_per_rule = bits_per_class + math.log2(max_rules_per_class)
    target_specified_bit_rate = target_specified_bits_per_rule / max_bits_per_rule
    assert 0 <= target_specified_bit_rate <= 1, target_specified_bit_rate

    wildcard_rate = 1 - target_specified_bit_rate
    mutation_rate = 1 / input_size

    algorithm = xcs.XCSAlgorithm()
    algorithm.max_population_size = max_rules
    algorithm.wildcard_probability = wildcard_rate  # 1 - 2 * math.log2(2 * 10) / 28 ** 2#1 - 10 / 28 ** 2
    algorithm.mutation_probability = mutation_rate
    algorithm.ga_threshold = 1
    # algorithm.deletion_threshold =
    algorithm.discount_factor = 0
    algorithm.do_ga_subsumption = True
    algorithm.do_action_set_subsumption = True
    algorithm.condition_type = IntervalCondition
    # linear: r=0.17560, e=-0.03687
    # scatter: r=0.21700, e=-0.01959
    # interval: r=0.19120, e=0.03567
    # blend: r=0.23600, e=0.00240
    # swap: r=0.16100, e=-0.05280
    # sort: r=0.30530, e=0.11110
    # interval+blend+sort: r=0.15710, e=-0.06727
    algorithm.crossover_params = dict(linear=False, scatter=False, interval=True, blend=True, swap=False, sort=True)

    train_scenario = scenarios.EMNIST('mnist', train=True)
    test_scenario = scenarios.EMNIST('mnist', train=False)

    model = algorithm.new_model(train_scenario)

    train_scenario = scenarios.ScenarioObserver(train_scenario, model, scenario_name='training')
    test_scenario = scenarios.ScenarioObserver(test_scenario, model, scenario_name='testing')

    for epoch in range(epochs):
        logger.info("Starting epoch %d", epoch + 1)

        if epoch:
            train_scenario.reset(reset_stats=True)
            test_scenario.reset(reset_stats=True)

        algorithm.exploration_probability = 1.0
        train_start_time = time.time()
        model.run(train_scenario, learn=True)
        train_end_time = time.time()
        logger.info("Total train time for epoch %d: %.5f seconds",
                    epoch + 1, train_end_time - train_start_time)
        logger.info("Total train reward: %.5f", train_scenario.total_reward)
        # if max(rule.numerosity for rule in model) == 1:
        #     algorithm.max_population_size *= 2

        algorithm.exploration_probability = 0.0
        test_start_time = time.time()
        model.run(test_scenario, learn=False)
        test_end_time = time.time()
        logger.info("Total test time for epoch %d: %.5f seconds",
                    epoch + 1, test_end_time - test_start_time)
        logger.info("Total test reward: %.5f", test_scenario.total_reward)

    return model


if __name__ == '__main__':
    test_mnist()
