import numpy as np
from collections import defaultdict
from sympy.ntheory import factorint
from itertools import product


def find_max_number_of_factors(n, l):
    max_number_of_factors = 0
    n_plus = n
    factors = None
    for i in range(n, l):
        d = factorint(i)
        number_of_factors = sum(list(d.values()))
        if number_of_factors > max_number_of_factors:
            max_number_of_factors = number_of_factors
            n_plus = i
            factors = dict(d)
    return n_plus, factors


def above_minimum_factors(factors, minimum_factors):
    not_enough = False
    for f, p in minimum_factors.items():
        if f not in factors:
            not_enough = True
        elif factors[f] < p:
            not_enough = True
        if not_enough:
            return False
    return True


def get_evaluation_conf(n,
        vergence_error_min=-8,
        vergence_error_max=8,
        stimulus_min=0,
        stimulus_max=20,
        depth_min=0.5,
        depth_max=5,
        speed_min=0,
        speed_max=1.125,
        direction_min=0,
        direction_max=2 * np.pi,
        tilt_on=True,
        pan_on=True,
        vergence_on=True,
        cyclo_on=True,
        priority=["stimulus", "vergence_errors", "depths", "speeds", "directions"]):
    # some dimensions have a fixed number of sample points and predetermined values
    priority = list(priority)
    sample_points = {}
    if tilt_on and pan_on:
        sample_points["directions"] = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        priority.remove("directions")
    elif tilt_on and not pan_on:
        sample_points["directions"] = [np.pi / 2, -np.pi / 2]
        priority.remove("directions")
    elif not tilt_on and pan_on:
        sample_points["directions"] = [0.0, np.pi]
        priority.remove("directions")
    elif not tilt_on and not pan_on:
        sample_points["directions"] = [0.0]
        sample_points["speeds"] = [0.0]
        priority.remove("directions")
        priority.remove("speeds")
    # print("sample_points = ")
    # print(sample_points)
    # print("\n")
    # print("priority = ")
    # print(priority)
    # print("\n")
    # this implies that we can factor the number of samples that we want with
    # the number of fixed sample points
    minimum_factors = defaultdict(int)
    for dimension, anchor in sample_points.items():
        if len(anchor) > 1:
            d = factorint(len(anchor))
            for factor, power in d.items():
                minimum_factors[factor] += power
    # print("minimum_factors = ")
    # print(minimum_factors)
    # print("\n")
    # find the smallest number > n which has the maximum number of factors and more factors than minimum_factors
    n_plus, factors = find_max_number_of_factors(n, n + 10)
    offset = 10
    while not above_minimum_factors(factors, minimum_factors):
        # print(offset)
        n_plus, factors = find_max_number_of_factors(n + offset, n + offset + 10)
        offset += 10
    # print(n_plus)
    # print("factors = ")
    # print(factors)
    # print("\n")
    # substracting minimum factors from factors (already assigned)
    for f, p in minimum_factors.items():
        factors[f] -= p
    # assign the factors to the dimension of which we take the product
    n_sample_points = {
        "stimulus": 1,
        "vergence_errors": 1,
        "depths": 1,
        "speeds": 1,
        "directions": 1,
    }
    for dimension, samples in sample_points.items():
        n_sample_points[dimension] = len(samples)
    while len(factors):
        for dimension in priority:
            if len(factors):
                max_factor = max(factors.keys())
                n_sample_points[dimension] *= max_factor
                factors[max_factor] -= 1
                if factors[max_factor] == 0:
                    factors.pop(max_factor)
                # print(factors)
                # print(n_sample_points)
            else:
                break
    if "vergence_errors" not in sample_points:
        sample_points["vergence_errors"] = np.linspace(vergence_error_min, vergence_error_max, n_sample_points["vergence_errors"])
    if "stimulus" not in sample_points:
        sample_points["stimulus"] = np.arange(n_sample_points["stimulus"])
    if "depths" not in sample_points:
        sample_points["depths"] = np.linspace(depth_min, depth_max, n_sample_points["depths"])
    if "speeds" not in sample_points:
        sample_points["speeds"] = np.linspace(speed_min, speed_max, n_sample_points["speeds"])
    if "directions" not in sample_points:
        sample_points["directions"] = np.linspace(direction_min, direction_max, n_sample_points["directions"])
    # print("n_sample_points = ")
    # print(n_sample_points)
    # print("\n")
    # print("sample_points = ")
    # print(sample_points)
    # print("\n")

    sorted_keys = sorted(sample_points)
    conf_dtype = np.dtype([(key, np.int32 if key in ["stimulus"] else np.float32) for key in sorted_keys])
    all_anchors = [sample_points[key] for key in sorted_keys]
    conf = np.zeros(n, dtype=conf_dtype)
    # for stuff in product(*all_anchors):
    #     print(stuff)

    for i, stuff in zip(range(n), product(*all_anchors)):
        conf[i] = stuff
    # print(conf, conf.dtype)
    return conf, n_sample_points



if __name__ == '__main__':
    get_evaluation_conf(400)
    # get_evaluation_conf(400, tilt_on=False)
    # get_evaluation_conf(400, pan_on=False)
    # get_evaluation_conf(400, tilt_on=False, pan_on=False)
