from unittest import TestCase
import csv
import numpy as np
import qrough.roughness as qr


def coherent_state_checker(_):
    return 1 / np.sqrt(6)


def squeezed_state_checker(z):
    t1 = np.exp(z) / (np.exp(2 * z) + 1)
    t2 = 4 * np.exp(z) / np.sqrt((np.exp(2 * z) + 2) * (2 * np.exp(2 * z) + 1))
    return np.sqrt(1 + t1 - t2)


def thermal_state_checker(nmed):
    den = 2 * (nmed + 1) * (2 * nmed + 1) * (4 * nmed + 3)
    return 1 / np.sqrt(den)


def even_cat_state_checker(q0):
    x = np.exp(-q0 ** 2)
    a1 = 7 / 12
    a2 = 5 / 6
    r2 = (a1 * (1 + x ** 2) + a2 * x - (2 / 3) * (x ** (2 / 3) + x ** (4 / 3))) / (1 + x) ** 2
    return np.sqrt(r2)


def odd_cat_state_checker(q0):
    x = np.exp(-q0 ** 2)
    a1 = 7 / 12
    a2 = 1 / 6
    r2 = (a1 * (1 + x ** 2) + a2 * x - (2 / 3) * (x ** (2 / 3) + x ** (4 / 3))) / (1 - x) ** 2
    return np.sqrt(r2)


test_functions = {
    'coherent_states': coherent_state_checker,
    'squeezed_states': squeezed_state_checker,
    'thermal_states': thermal_state_checker,
    'even_cat_states': even_cat_state_checker,
    'odd_cat_states': odd_cat_state_checker
}


def states_info(filename):
    info = {}
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            info[row['name']] = float(row['parameter'])
    return info


def basic_states_test_loader(base_name):
    basic_info = states_info(base_name + ".csv")
    states = np.load(base_name + ".npz")
    checker = test_functions[base_name]
    testing_values = []
    for fname, rho in states.items():
        par_value = basic_info[fname]
        target_value = checker(par_value)
        r = qr.rough(rho)
        testing_values.append((par_value, target_value, r))
    return testing_values


class TestRoughness(TestCase):

    def test_coherent_state_roughness(self):
        testing_values = basic_states_test_loader('coherent_states')
        for par_value, target_value, r in testing_values:
            msg = f'\nroughness of coherent state failed for {par_value}'
            self.assertAlmostEqual(r, target_value, places=13, msg=msg)

    def test_squeezed_state_roughness(self):
        testing_values = basic_states_test_loader('squeezed_states')
        for par_value, target_value, r in testing_values:
            msg = f'\nroughness of squeezed state failed for {par_value}'
            self.assertAlmostEqual(r, target_value, places=13, msg=msg)

    def test_thermal_state_roughness(self):
        testing_values = basic_states_test_loader('thermal_states')
        for par_value, target_value, r in testing_values:
            msg = f'\nroughness of thermal state failed for {par_value}'
            self.assertAlmostEqual(r, target_value, places=13, msg=msg)

    def test_even_cat_state_roughness(self):
        testing_values = basic_states_test_loader('even_cat_states')
        for par_value, target_value, r in testing_values:
            msg = f'\nroughness of even cat state failed for {par_value}'
            self.assertAlmostEqual(r, target_value, places=13, msg=msg)

    def test_odd_cat_state_roughness(self):
        testing_values = basic_states_test_loader('odd_cat_states')
        for par_value, target_value, r in testing_values:
            msg = f'\nroughness of even cat state failed for {par_value}'
            self.assertAlmostEqual(r, target_value, places=13, msg=msg)
