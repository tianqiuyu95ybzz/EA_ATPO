import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_experts import evaluate, finalize_a


def item(sample, case, base, action, candidate, score=0.8):
    return {'sample_id': sample, 'case_id': case, 'gold_status': 'APPROVED',
            'base_answer': 'raw Base 答案\n', 'base_correct': base,
            'expert_output': json.dumps({'action': action, 'answer': 'corrected' if action == 'REVISE' else 'ignored'}),
            'expert_correct': candidate, 'decision_score': score,
            'target_action': 'KEEP' if base else 'REVISE',
            'correctness_method': 'human_adjudicated', 'risk_level': 'HIGH'}


class PairedEvaluationTest(unittest.TestCase):
    def test_keep_and_abstain_copy_base_verbatim(self):
        base = ' 原始答案\n'
        for action in ('KEEP', 'ABSTAIN'):
            result = json.dumps({'action': action, 'answer': 'model changed this'})
            self.assertEqual(finalize_a(base, result), (action, base))
        result = json.dumps({'action': 'REVISE', 'answer': 'changed'})
        self.assertEqual(finalize_a(base, result, False), ('ABSTAIN', base))
        self.assertEqual(finalize_a(base, result, True), ('REVISE', 'changed'))

    def test_four_cells_and_distinct_oracles(self):
        records = [item('1', 'case-a', True, 'KEEP', None),
                   item('2', 'case-a', True, 'REVISE', False),
                   item('3', 'case-b', False, 'REVISE', True),
                   item('4', 'case-b', False, 'ABSTAIN', None)]
        result, cases, curve = evaluate(records, 'A', 0.0, 300, 19)
        cells = result['comparison']
        self.assertEqual([cells[k] for k in ('base_correct_expert_correct', 'base_correct_expert_wrong',
                                             'base_wrong_expert_correct', 'base_wrong_expert_wrong')], [1, 1, 1, 1])
        self.assertEqual(cells['correction_rate'], 0.5)
        self.assertEqual(cells['degradation_rate'], 0.5)
        self.assertEqual(cells['net_gain_count'], 0)
        self.assertEqual(result['oracle_expert_accuracy_theoretical'], 1.0)
        self.assertEqual(result['oracle_selector_accuracy_theoretical'], 0.75)
        self.assertEqual(result['bootstrap_unit'], 'case_id')
        self.assertEqual(len(cases), 2)
        self.assertEqual(len(curve), 21)
        self.assertEqual(result['high_risk_final_errors'], 2)

    def test_independent_expert_and_selection_are_separate(self):
        records = [item('1', 'case-a', True, 'KEEP', None, score=0.1),
                   item('2', 'case-b', False, 'REVISE', True, score=0.9)]
        for r, correctness in zip(records, (False, True)):
            r['expert_output'] = 'expert raw answer'
            r['expert_correct'] = correctness
        result, _, curve = evaluate(records, 'B', 0.5, 200, 42)
        self.assertEqual(result['comparison']['expert_accuracy'], 0.5)
        self.assertEqual(result['final_policy']['expert_accuracy'], 1.0)
        self.assertEqual(curve[10]['expert_selection_rate'], 0.5)

    def test_pending_and_missing_adjudication_fail_closed(self):
        record = item('1', 'case-a', False, 'REVISE', True)
        record['gold_status'] = 'PENDING'
        with self.assertRaises(ValueError):
            evaluate([record], 'A', 0.0, 10, 1)
        record['gold_status'] = 'APPROVED'
        record['correctness_method'] = 'model_guess'
        with self.assertRaises(ValueError):
            evaluate([record], 'A', 0.0, 10, 1)

    def test_malformed_action_is_counted_and_falls_back(self):
        record = item('1', 'case-a', True, 'KEEP', None)
        record['expert_output'] = '{invalid json'
        result, _, _ = evaluate([record], 'A', 0.0, 25, 1)
        self.assertEqual(result['invalid_output_count_including_ambiguous'], 1)
        self.assertEqual(result['action_counts_including_ambiguous']['ABSTAIN'], 1)
        self.assertEqual(result['comparison']['expert_accuracy'], 1.0)

    def test_empty_independent_output_is_counted_wrong(self):
        record = item('1', 'case-a', True, 'REVISE', False)
        record.update({'expert_output': '', 'correctness_method': 'invalid_output'})
        result, _, _ = evaluate([record], 'B', 0.0, 25, 1)
        self.assertEqual(result['invalid_output_count_including_ambiguous'], 1)
        self.assertEqual(result['comparison']['expert_accuracy'], 0.0)


if __name__ == '__main__':
    unittest.main()
