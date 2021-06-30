# For testing that all the main files run successfully
import unittest
import numpy as np


class CommonFunctionsUnittests(unittest.TestCase):

    def test_IR_additivity(self):
        from common.apm_functions import IR_additivity
        ls_ic_br = [(0.02, 12), (0.04, 24), (0.075, 100)]

        func_res = IR_additivity(ls_ic_br)
        IR_1_sq = ls_ic_br[0][0] ** 2 * ls_ic_br[0][1]
        IR_2_sq = ls_ic_br[1][0] ** 2 * ls_ic_br[1][1]
        IR_3_sq = ls_ic_br[2][0] ** 2 * ls_ic_br[2][1]

        self.assertEqual(np.sqrt(sum([IR_1_sq, IR_2_sq, IR_3_sq])), func_res)

    def test_uncertain_cashflows(self):
        from common.cashflows import discounted_expected_uncertain_cashflows, \
            discounted_expected_uncertain_cashflows_risk_adjusted

        probabilities, uncertain_cashflows, t = np.array([0.5, 0.5]), np.array([49, 53]), 1 / 12
        val_multiplers = np.array([1.38, 0.62])

        without_risk = discounted_expected_uncertain_cashflows(probabilities, uncertain_cashflows, t)

        with_risk = discounted_expected_uncertain_cashflows_risk_adjusted(probabilities, uncertain_cashflows,
                                                                          val_multiplers, t)

        self.assertTrue(np.allclose([50.75295741344933, 49.99663883238616], [without_risk, with_risk]))
