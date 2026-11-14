import copy
import json
import tempfile
import unittest
from pathlib import Path

from field_transport_lab import transport
from field_transport_lab.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def load(name='transport.json'):
    return read_json(ROOT / 'fixtures' / name)


def by_id(result, identifier):
    for item in result['estimates']:
        if item['id'] == identifier:
            return item
    raise AssertionError('missing estimate ' + identifier)


def scenario(result, identifier):
    for item in result['sensitivity']['scenarios']:
        if item['id'] == identifier:
            return item
    raise AssertionError('missing scenario ' + identifier)


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.document = load()
        self.result, self.artifact = transport.fit(self.document)

    def test_recruitment_excluded_hardware_is_not_estimated(self):
        coverage = self.result['target_coverage']
        self.assertEqual(self.result['target_coverage']['columns_used'], ['tier', 'task'])
        self.assertAlmostEqual(coverage['supported_share'], 2 / 3)
        unsupported = {tuple(item['key']): item for item in coverage['unsupported_cells']}
        self.assertIn(('entry', 'summary'), unsupported)
        self.assertEqual(unsupported[('entry', 'summary')]['reason'], 'recruitment_excluded')

    def test_entry_target_mass_keeps_point_estimate_out_of_population(self):
        primary = by_id(self.result, 'primary_transport')
        self.assertEqual(primary['population'], 'supported_region_only')
        self.assertEqual(primary['supported_share'], 2 / 3)
        self.assertIsNone(by_id(self.result, 'whole_population_bounds')['estimate'])
        low, high = by_id(self.result, 'whole_population_bounds')['interval']
        self.assertLess(low, primary['estimate'])
        self.assertGreater(high, primary['estimate'])

    def test_same_encoding_different_meaning_is_excluded(self):
        self.assertIn('network', self.result['target_coverage']['excluded_columns'])
        marginal = self.result['balance']['marginals']['network']
        self.assertFalse(marginal['used_for_transport'])
        self.assertEqual(marginal['levels_seen_in_source'], ['normal', 'weak'])

    def test_meaning_mismatch_scenario_moves_the_answer(self):
        naive = scenario(self.result, 'naive_meaning_mismatch')
        self.assertIsNotNone(naive['estimate'])
        self.assertLess(naive['supported_share'], 2 / 3)
        self.assertNotAlmostEqual(naive['estimate'],
                                  by_id(self.result, 'primary_transport')['estimate'])

    def test_at_least_one_no_transport_reference(self):
        self.assertIsNotNone(by_id(self.result, 'lab_reference')['estimate'])
        self.assertIsNotNone(scenario(self.result, 'naive_lab_pooled')['estimate'])
        self.assertNotAlmostEqual(by_id(self.result, 'lab_reference')['estimate'],
                                  by_id(self.result, 'primary_transport')['estimate'])

    def test_model_never_aggregates_unsupported_cells(self):
        model = by_id(self.result, 'outcome_model_scenario')
        self.assertEqual(model['population'], 'supported_region_only')
        extrapolation = self.result['sensitivity']['extrapolation']
        withheld = [item for item in extrapolation['predictions']
                    if not item['used_in_estimate']]
        self.assertTrue(withheld)
        self.assertTrue(all(item['key'][0] == 'entry' for item in withheld))
        self.assertGreater(extrapolation['withheld_target_weight'], 0)
        self.assertLess(abs(model['estimate'] - by_id(self.result, 'primary_transport')['estimate']),
                        1.0)

    def test_excluded_field_shows_unadjustable_background_difference(self):
        background = self.result['assumptions']['background_difference_not_adjustable']
        self.assertEqual(background['status'], 'present')
        field = background['fields'][0]
        self.assertEqual(field['field'], 'network')
        self.assertEqual(field['target_share_within_supported'], {'weak': 1.0})
        self.assertGreater(field['source_share_within_supported']['normal'], 0.5)
        per_cell = {tuple(item['key']): item for item in field['per_supported_cell']}
        self.assertEqual(per_cell[('high', 'booking')]['source_share'], {'normal': 1.0})
        self.assertEqual(per_cell[('mid', 'booking')]['source_share'],
                         {'normal': 2 / 3, 'weak': 1 / 3})

    def test_whole_population_bounds_cover_the_supported_estimate(self):
        estimate = by_id(self.result, 'primary_transport')['estimate']
        low, high = by_id(self.result, 'whole_population_bounds')['interval']
        self.assertAlmostEqual(low, estimate * 2 / 3)
        self.assertAlmostEqual(high, estimate * 2 / 3 + 1 / 3)


class WeightAndBalanceTests(unittest.TestCase):
    def setUp(self):
        self.document = load()
        self.result, _ = transport.fit(self.document)

    def test_weight_ratio_reported_and_trimming_bounds_it(self):
        balance = self.result['balance']
        self.assertAlmostEqual(balance['max_weight_ratio'], 2.5)
        self.assertAlmostEqual(balance['min_weight_ratio'], 0.625)
        primary = by_id(self.result, 'primary_transport')['estimate']
        trimmed = by_id(self.result, 'trimmed_transport')['estimate']
        self.assertNotAlmostEqual(primary, trimmed)
        caps = [item for item in self.result['sensitivity']['scenarios']
                if item['id'].startswith('weight_cap_')]
        self.assertGreaterEqual(len(caps), 2)
        cap_one = scenario(self.result, 'weight_cap_1')
        self.assertAlmostEqual(cap_one['estimate'], 0.9047619047619049)

    def test_effective_sample_below_usable_rows(self):
        effective = self.result['sensitivity']['effective_sample_size']
        self.assertLess(effective['effective_rows_after_weighting'],
                        effective['source_rows_in_support'])
        self.assertEqual(effective['source_rows_in_support'], 180)

    def test_extreme_cap_larger_than_ratio_is_inactive(self):
        result, _ = transport.fit(self.document, trim_cap=50)
        self.assertFalse(by_id(result, 'trimmed_transport')['trimmed'])
        self.assertAlmostEqual(by_id(result, 'trimmed_transport')['estimate'],
                               by_id(result, 'primary_transport')['estimate'])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.document = load()
        self.result, _ = transport.fit(self.document)

    def test_only_holdout_is_used_for_validation(self):
        validation = self.result['validation']
        self.assertTrue(validation['holdout_only'])
        self.assertEqual(validation['holdout_rows'], 16)
        self.assertEqual(validation['development_rows_excluded_from_validation'], 8)
        self.assertFalse(validation['used_for_fit'])
        self.assertFalse(validation['random_truth'])

    def test_selection_bias_and_transport_error_are_separate(self):
        validation = self.result['validation']
        self.assertIn('transport_error', validation)
        self.assertIn('selection_bias', validation)
        for cell in validation['per_cell']:
            self.assertIsNotNone(cell['transport_gap'])
            self.assertIn('selection_bias_where_known', cell)
        self.assertEqual(validation['transport_error']['cells_compared'], 2)
        self.assertAlmostEqual(
            validation['transport_error']['mean_gap_holdout_minus_source_supported'], 0.0)

    def test_missing_selection_probability_is_not_invented(self):
        cells = {tuple(cell['key']): cell for cell in self.result['validation']['per_cell']}
        high = cells[('high', 'booking')]
        self.assertEqual(high['selection_probability_known_rows'], 4)
        self.assertIsNotNone(high['ipw_rate_where_known'])
        self.assertEqual(high['rows'], 8)

    def test_holdout_outcomes_are_never_read_during_fit(self):
        perturbed = copy.deepcopy(self.document)
        for row in perturbed['validation']:
            row['completed'] = not row['completed']
        result, artifact = transport.fit(perturbed)
        original, original_artifact = transport.fit(self.document)
        self.assertEqual([item['estimate'] for item in result['estimates']],
                         [item['estimate'] for item in original['estimates']])
        self.assertEqual(artifact['fingerprint_source'],
                         original_artifact['fingerprint_source'])

    def test_dropping_holdout_keeps_estimates(self):
        document = copy.deepcopy(self.document)
        document['validation'] = [row for row in document['validation']
                                  if row['split'] != 'holdout']
        result, _ = transport.fit(document)
        original, _ = transport.fit(self.document)
        self.assertEqual([item['estimate'] for item in result['estimates']],
                         [item['estimate'] for item in original['estimates']])
        self.assertEqual(result['validation']['status'], 'no_holdout')


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.document = load()
        self.result, self.artifact = transport.fit(self.document)
        self.artifact = json.loads(json.dumps(self.artifact, allow_nan=False))

    def test_artifact_round_trips_through_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'artifact.json'
            write_json(path, self.artifact)
            loaded = read_json(path)
            applied = transport.apply(loaded, self.document)
        self.assertAlmostEqual(applied['estimates'][0]['estimate'],
                               by_id(self.result, 'primary_transport')['estimate'])

    def test_apply_to_new_region_does_not_need_outcomes(self):
        new_document = copy.deepcopy(self.document)
        new_document['target'] = [row for row in new_document['target']
                                  if row['tier'] != 'high']
        for row in new_document['target']:
            row['row_id'] = 'n-' + row['row_id']
            row['population_weight'] = 10
        applied = transport.apply(self.artifact, new_document)
        self.assertAlmostEqual(applied['target_coverage']['supported_share'], 0.5)
        high = [cell for cell in applied['target_coverage']['cells']
                if cell['key'][0] == 'high']
        self.assertEqual(high, [])
        self.assertEqual(applied['estimates'][0]['estimate'], 0.8055555555555556)
        self.assertFalse(applied['artifact']['recomputed_from_target_outcomes'])

    def test_apply_rejects_meaning_change_in_new_snapshot(self):
        new_document = copy.deepcopy(self.document)
        for row in new_document['target']:
            row['measurement_version'] = 'field-v9'
        with self.assertRaises(ValueError):
            transport.apply(self.artifact, new_document)

    def test_apply_rejects_missing_version_in_new_snapshot(self):
        new_document = copy.deepcopy(self.document)
        for row in new_document['target']:
            row['measurement_version'] = None
        with self.assertRaises(ValueError):
            transport.apply(self.artifact, new_document)

    def test_supported_cell_missing_in_new_region_is_unsupported(self):
        new_document = copy.deepcopy(self.document)
        new_document['target'] = [row for row in new_document['target']
                                  if row['tier'] != 'mid']
        applied = transport.apply(self.artifact, new_document)
        cells = {tuple(cell['key']): cell for cell in applied['target_coverage']['cells']}
        self.assertNotIn(('mid', 'booking'), cells)
        self.assertAlmostEqual(applied['target_coverage']['supported_share'], 0.5)
        self.assertAlmostEqual(applied['estimates'][0]['estimate'], 0.9444444444444444)


class MeasurementAndInputTests(unittest.TestCase):
    def test_unversioned_field_is_usable_but_flagged(self):
        document = load()
        document['measurement_versions'] = [
            entry for entry in document['measurement_versions']
            if entry['field'] != 'network']
        result, _ = transport.fit(document)
        self.assertIn('network', result['target_coverage']['columns_used'])

    def test_missing_covariate_target_weight_is_unsupported(self):
        document = load()
        for row in document['target'][:5]:
            row['task'] = None
        result, _ = transport.fit(document)
        reasons = result['target_coverage']['unsupported_weight_by_reason']
        self.assertIn('missing_covariate', reasons)
        self.assertGreater(reasons['missing_covariate'], 0)
        scenario_item = scenario(result, 'missing_covariate')
        self.assertIsNotNone(scenario_item)
        self.assertEqual(scenario_item['target_missing_rows']['task'], 5)

    def test_duplicate_row_ids_rejected(self):
        document = load()
        document['source'][1]['row_id'] = document['source'][0]['row_id']
        with self.assertRaises(ValueError):
            transport.fit(document)

    def test_negative_population_weight_rejected(self):
        document = load()
        document['target'][0]['population_weight'] = -1
        with self.assertRaises(ValueError):
            transport.fit(document)

    def test_insufficient_source_cell_is_unsupported(self):
        document = load()
        result, _ = transport.fit(document, min_source_cell=50)
        supported = [cell for cell in result['target_coverage']['cells']
                     if cell['status'] == 'supported']
        self.assertEqual({tuple(cell['key']) for cell in supported},
                         {('high', 'booking')})
        reasons = result['target_coverage']['unsupported_weight_by_reason']
        self.assertIn('insufficient_source_support', reasons)

    def test_partial_row_version_is_not_silently_assumed(self):
        document = load()
        document['source'][0]['measurement_version'] = None
        result, _ = transport.fit(document)
        self.assertIn('network', result['target_coverage']['excluded_columns'])
        # network was already excluded for meaning; make a comparable field
        # unverifiable instead: rename the task version entry.
        document = load()
        document['source'][0]['measurement_version'] = 'lab-v9'
        document['measurement_versions'].append(
            {'field': 'network', 'version': 'lab-v9', 'meaning': '接入信号等级'})
        result, _ = transport.fit(document)
        self.assertIn('network', result['target_coverage']['excluded_columns'])

    def test_next_region_fixture_keeps_unsupported_hardware(self):
        document = load('next_region.json')
        _, artifact = transport.fit(load())
        applied = transport.apply(json.loads(json.dumps(artifact)), document)
        statuses = {tuple(cell['key']): cell['reason']
                    for cell in applied['target_coverage']['cells']}
        self.assertEqual(statuses[('entry', 'summary')], 'recruitment_excluded')
        self.assertEqual(statuses[('entry', 'booking')], 'recruitment_excluded')
        self.assertLess(applied['target_coverage']['supported_share'], 1)
        self.assertIsNotNone(applied['estimates'][0]['estimate'])


class CliTests(unittest.TestCase):
    def test_module_cli_fit_and_apply(self):
        from field_transport_lab.transport import _main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            artifact = Path(directory) / 'artifact.json'
            self.assertEqual(_main(['fit', '--input', str(ROOT / 'fixtures/transport.json'),
                                    '--output', str(output),
                                    '--artifact', str(artifact)]), 0)
            self.assertTrue(output.exists())
            self.assertTrue(artifact.exists())
            applied = Path(directory) / 'applied.json'
            self.assertEqual(_main(['apply', '--input', str(ROOT / 'fixtures/transport.json'),
                                    '--output', str(applied),
                                    '--artifact', str(artifact)]), 0)
            result = read_json(applied)
            self.assertIn('target_coverage', result)
            self.assertIn('estimates', result)

    def test_cli_reports_bad_input_with_nonzero_status(self):
        from field_transport_lab.transport import _main
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / 'bad.json'
            bad.write_text('{"columns": ["tier"]}', encoding='utf-8')
            output = Path(directory) / 'result.json'
            self.assertEqual(_main(['fit', '--input', str(bad), '--output', str(output)]), 2)


if __name__ == '__main__':
    unittest.main()
