import unittest
from demo import synthetic_data,analyse,validate


class PaymentAnalysisChecks(unittest.TestCase):
    def test_known_synthetic_effect_is_detected(self):
        data,groups,lpm,logit,ame=analyse(synthetic_data())
        self.assertEqual(len(data),800)
        self.assertGreater(groups.loc[1,'mean']-groups.loc[0,'mean'],.08)
        self.assertGreater(lpm.params['Default'],.08)
        self.assertTrue(.05<ame<.3)

    def test_only_analysis_columns_are_retained(self):
        frame=synthetic_data();frame['Email']='synthetic@example.invalid'
        self.assertNotIn('Email',validate(frame).columns)

    def test_single_group_input_is_rejected(self):
        frame=synthetic_data();frame['Default']=0
        with self.assertRaises(ValueError):validate(frame)


if __name__=='__main__':unittest.main()
