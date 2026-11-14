"""Transport analysis: support, balance, estimates, sensitivity, validation.

Design boundary
---------------
The estimator is a finite-population transport of the source outcome rate to the
*measurable* target population.  Only target cells with source support are in
the estimand; target cells the source never covered are reported as unsupported
and carry their population weight into a separate whole-population bound.

Causal assumption that is checkable here:
  * exchangeability of the source-outcome law across the covariate levels that
    both sides cover, in the sense that the source rate for a covered cell is
    the target rate for that cell.

Causal assumptions that are NOT checkable, only stress-tested (``sensitivity``):
  * outcome-model functional form (cell means vs additive logit);
  * unmeasured confounding of the field validation selection (only informative
    where ``selected_probability`` is recorded and can be used as a sampling
    weight);
  * measurement equivalence of a covariate whose meaning differs even though
    its encoding is identical.

No target or validation outcome is used to fit the artifact.
"""

import hashlib
import json
import math
import sys

import numpy as np
from scipy.special import expit

from .support import (assess_measurement, build_cell_table, cell_label,
                      classify_target_cells, finite_weight, levels_from,
                      summarize_coverage)


ARTIFACT_FORMAT = 'field_transport_lab/transport-artifact'
ARTIFACT_VERSION = 1
DEFAULT_MIN_SOURCE_CELL = 8
DEFAULT_TRIM_CAP = 2.0
DEFAULT_Z = 1.96


# ---------------------------------------------------------------------------
# input extraction and validation


def _columns(document):
    columns = document.get('columns')
    if not isinstance(columns, list) or not columns or \
            not all(isinstance(name, str) for name in columns):
        raise ValueError('columns 必须是非空字符串数组')
    if len(set(columns)) != len(columns):
        raise ValueError('columns 有重复字段')
    return list(columns)


def _rows(document, name, require_outcome, weight_field):
    rows = document.get(name)
    if not isinstance(rows, list):
        raise ValueError(name + ' 必须是数组')
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(name + ' 第 ' + str(index) + ' 行不是对象')
        row_id = row.get('row_id')
        if row_id is None:
            raise ValueError(name + ' 第 ' + str(index) + ' 行缺少 row_id')
        if str(row_id) in seen:
            raise ValueError(name + ' 行编号重复：' + str(row_id))
        seen.add(str(row_id))
        if weight_field is not None:
            if 'population_weight' not in row:
                raise ValueError(name + ' 行 ' + str(row_id) + ' 缺少 population_weight')
            finite_weight(row['population_weight'], 'population_weight')
        if require_outcome:
            if 'completed' not in row:
                raise ValueError(name + ' 行 ' + str(row_id) + ' 缺少 completed')
            if row['completed'] is not None and not isinstance(row['completed'], bool):
                raise ValueError(name + ' 行 ' + str(row_id) + ' 的 completed 必须是布尔值或 null')
    return rows


def _recruitment(document):
    recruitment = document.get('recruitment') or {}
    if not isinstance(recruitment, dict):
        raise ValueError('recruitment 必须是对象')
    excluded = recruitment.get('excluded_tiers', []) or []
    if not isinstance(excluded, list):
        raise ValueError('recruitment.excluded_tiers 必须是数组')
    return {'excluded_tiers': [str(value) for value in excluded],
            'required_network': recruitment.get('required_network')}


def _fingerprint(document, columns):
    payload = {
        'columns': columns,
        'source': [[row.get('row_id')] + [row.get(column) for column in columns] +
                   [row.get('completed'), row.get('measurement_version'),
                    row.get('device_id')] for row in document['source']],
        'measurement_versions': document.get('measurement_versions'),
        'recruitment': _recruitment(document),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _number(value):
    text = ('%.2f' % value).rstrip('0').rstrip('.')
    return text if text else '0'


# ---------------------------------------------------------------------------
# weights and estimates


def _share_tables(source_table, target_table, supported):
    keys = [tuple(record['key']) for record in supported]
    target_weight = {key: target_table[key]['weight'] for key in keys}
    source_rows = {key: source_table[key]['rows'] for key in keys}
    target_total = sum(target_weight.values())
    source_total = sum(source_rows.values())
    if target_total <= 0 or source_total <= 0:
        raise ValueError('支持区域的目标权重或来源样本必须为正')
    return ({key: value / target_total for key, value in target_weight.items()},
            {key: value / source_total for key, value in source_rows.items()})


def _cap_weights(raw, target_share, source_share, cap):
    """Cap the target/source weight ratio, then renormalize.

    ``cap >= max(raw)`` leaves the target shares untouched; ``cap == 1`` falls
    back to the source composition.  This is a stress test on how much of the
    estimate is driven by a few high-weight cells.
    """
    shares = {key: source_share[key] * min(raw[key], cap) for key in raw}
    total = sum(shares.values())
    if total <= 0:
        return dict(target_share)
    return {key: value / total for key, value in shares.items()}


def _weighted_estimate(source_table, weights):
    estimate = 0.0
    used = 0
    maximum = 0.0
    for key, weight in weights.items():
        if weight <= 0:
            continue
        cell = source_table[key]
        if cell['outcomes'] <= 0:
            raise ValueError('目标的某个支持格子在来源中没有可用结局')
        estimate += weight * (cell['successes'] / cell['outcomes'])
        used += 1
        maximum = max(maximum, weight)
    return {'estimate': estimate if used else None, 'cells_used': used,
            'max_cell_weight': maximum if used else None}


def _cluster_se(source_table, weights):
    """First-order sampling SE of the source cell rates, clustered by device.

    Each source row contributes w_k * (y_i - rate_k) / n_k; rows that share a
    device are summed before squaring, because a lab device supplies several
    rows and those rows are not independent evidence.  Target composition
    weights are treated as fixed, so this interval does NOT cover uncertainty
    in the target snapshot or in the exchangeability assumption itself.
    """
    influence = {}
    used = 0
    for key, weight in weights.items():
        if weight <= 0:
            continue
        cell = source_table[key]
        if cell['outcomes'] <= 0:
            continue
        rate = cell['successes'] / cell['outcomes']
        for device, stats in cell['device_stats'].items():
            if not stats['rows']:
                continue
            contribution = weight * (stats['successes'] - rate * stats['rows']) / cell['outcomes']
            influence[device] = influence.get(device, 0.0) + contribution
        used += 1
    if not used:
        return None
    return math.sqrt(sum(value * value for value in influence.values()))


def _wilson(successes, trials, z=DEFAULT_Z):
    if not trials:
        return None
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _normal_interval(estimate, se, z):
    if estimate is None or se is None or not math.isfinite(se):
        return None
    return [max(0.0, estimate - z * se), min(1.0, estimate + z * se)]


def _effective_rows(weights_by_key, source_table):
    values = []
    for key, weight in weights_by_key.items():
        rows = source_table[key]['rows']
        if rows:
            values.extend([weight / rows] * rows)
    if not values:
        return None
    values = np.array(values)
    denominator = float(np.dot(values, values))
    return float(values.sum() ** 2 / denominator) if denominator else None


# ---------------------------------------------------------------------------
# outcome model scenario


def _categorical_matrix(rows, columns):
    X = np.ones((len(rows), 1))
    blocks = []
    for column in columns:
        levels = sorted({row.get(column) for row in rows}, key=str)
        blocks.append(levels)
        width = len(levels) - 1
        if width:
            block = np.zeros((len(rows), width))
            for row_index, row in enumerate(rows):
                value = row.get(column)
                if value in levels[1:]:
                    block[row_index, levels[1:].index(value)] = 1.0
            X = np.hstack([X, block])
    return X, blocks


def _matrix_with_blocks(rows, columns, blocks):
    X = np.zeros((len(rows), 1 + sum(len(block) - 1 for block in blocks)))
    X[:, 0] = 1.0
    offset = 1
    for column, block in zip(columns, blocks):
        width = len(block) - 1
        if width:
            for row_index, row in enumerate(rows):
                value = row.get(column)
                if value in block[1:]:
                    X[row_index, offset + block[1:].index(value)] = 1.0
        offset += width
    return X


def _fit_logit(X, y, ridge=1e-8, iterations=100):
    beta = np.zeros(X.shape[1])
    hessian = np.eye(X.shape[1])
    for _ in range(iterations):
        probability = expit(X @ beta)
        variance = probability * (1 - probability) + 1e-12
        hessian = X.T @ (X * variance[:, None]) + ridge * np.eye(X.shape[1])
        gradient = X.T @ (y - probability) - ridge * beta
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    bread = np.linalg.inv(hessian)
    return beta, bread


def _outcome_model_scenario(source, target, supported, transport_columns, target_share, z):
    keys = {tuple(record['key']) for record in supported}
    try:
        X_source, blocks = _categorical_matrix(source, transport_columns)
        y = np.array([1.0 if row['completed'] else 0.0 for row in source])
        beta, bread = _fit_logit(X_source, y)
    except (np.linalg.LinAlgError, ValueError) as error:
        return ({'estimate': None, 'se': None},
                {'status': 'unavailable',
                 'reason': '结果模型无法拟合：' + type(error).__name__,
                 'predictions': []})

    supported_rows = [row for row in target
                      if tuple(row.get(column) for column in transport_columns) in keys]
    if not supported_rows:
        return ({'estimate': None, 'se': None},
                {'status': 'unavailable', 'reason': '支持区域没有目标行',
                 'predictions': []})
    X_target = _matrix_with_blocks(supported_rows, transport_columns, blocks)
    probability = expit(X_target @ beta)
    population_weight = np.array([float(row['population_weight']) for row in supported_rows])
    population_weight = population_weight / population_weight.sum()
    estimate = float(population_weight @ probability)

    devices = {}
    for index, row in enumerate(source):
        devices.setdefault(str(row.get('device_id', index)), []).append(index)
    meat = np.zeros((beta.size, beta.size))
    for indices in devices.values():
        indices = np.array(indices)
        residual = y[indices] - expit(X_source[indices] @ beta)
        score = X_source[indices].T @ residual
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    gradient = (population_weight[:, None] * probability[:, None] *
                (1 - probability[:, None])).T @ X_target
    gradient = np.asarray(gradient).reshape(-1)
    se = math.sqrt(max(float(gradient @ covariance @ gradient), 0.0))

    # The model can emit a number for any cell.  Only supported cells are ever
    # aggregated; unsupported predictions are listed and explicitly withheld.
    all_keys = sorted({tuple(row.get(column) for column in transport_columns)
                       for row in target}, key=lambda value: tuple(str(v) for v in value))
    predictions = []
    for key in all_keys:
        rows_here = [row for row in target
                     if tuple(row.get(column) for column in transport_columns) == key]
        X_here = _matrix_with_blocks(rows_here, transport_columns, blocks)
        predictions.append({
            'key': list(key),
            'status': 'supported' if key in keys else 'unsupported',
            'target_weight': float(sum(float(row['population_weight']) for row in rows_here)),
            'model_prediction': float(np.mean(expit(X_here @ beta))),
            'used_in_estimate': key in keys,
        })
    withheld = [item for item in predictions if not item['used_in_estimate']]
    return ({'estimate': estimate, 'se': se,
             'interval': _normal_interval(estimate, se, z)},
            {'status': 'computed',
             'reason': '模型能对未支持硬件层输出数值，但这些数值没有进入任何人群结论',
             'predictions': predictions,
             'withheld_unsupported_predictions': len(withheld),
             'withheld_target_weight': sum(item['target_weight'] for item in withheld)})


# ---------------------------------------------------------------------------
# sensitivity


def _measurement_mismatch_scenario(source, target, columns):
    """Run the whole pipeline with a column whose meaning differs.

    The encodings match, so nothing in the data stops this computation.  It is
    returned only as a sensitivity number: it quantifies what the report would
    have said if measurement equivalence had been silently assumed.
    """
    levels = levels_from(source, columns)
    source_table = build_cell_table(source, columns, outcome_field='completed')
    target_table = build_cell_table(target, columns, weight_field='population_weight')
    records = classify_target_cells(target_table, columns, source_table,
                                    DEFAULT_MIN_SOURCE_CELL, levels,
                                    {'excluded_tiers': [], 'required_network': None})
    supported = [record for record in records if record['status'] == 'supported']
    coverage = summarize_coverage(records)
    estimate = None
    if supported:
        target_total = sum(record['target_weight'] for record in supported)
        weights = {tuple(record['key']): record['target_weight'] / target_total
                   for record in supported}
        estimate = _weighted_estimate(source_table, weights)['estimate']
    return {
        'id': 'naive_meaning_mismatch',
        'type': 'measurement',
        'assumption': '错误假设：同名编码的字段在两边测量含义一致',
        'estimate': estimate,
        'supported_share': coverage['supported_share'],
        'difference_from_primary': None,
        'columns_used': list(columns),
        'note': '只用来衡量测量含义不一致带来的偏差；不作为结论，也不进入全人群边界。',
    }


def _marginals(source, target, columns, transport_columns):
    """Marginal covariate distributions, with excluded fields flagged."""
    supported_columns = set(transport_columns)
    result = {}
    for column in columns:
        source_total = len(source)
        target_total = sum(float(row['population_weight']) for row in target)
        counts = {}
        for row in source:
            counts.setdefault(str(row.get(column)), 0.0)
            counts[str(row.get(column))] += 1.0
        target_counts = {}
        for row in target:
            key = str(row.get(column))
            target_counts.setdefault(key, 0.0)
            target_counts[key] += float(row['population_weight'])
        seen = sorted({str(row.get(column)) for row in source
                       if row.get(column) is not None})
        result[column] = {
            'used_for_transport': column in supported_columns,
            'source_share': {key: value / source_total for key, value in sorted(counts.items())}
            if source_total else {},
            'target_share': {key: value / target_total for key, value in sorted(target_counts.items())}
            if target_total else {},
            'levels_seen_in_source': seen,
            'target_levels_without_source': sorted(
                {str(row.get(column)) for row in target if row.get(column) is not None} - set(seen)),
            'note': ('用于迁移' if column in supported_columns else
                     '测量含义不可比，仅列出边际，不用于权重或估计'),
        }
    return result


def _sensitivity(source, target, supported, columns, transport_columns, target_share,
                 source_share, source_table, coverage, trim_cap, primary_estimate):
    scenarios = []

    raw = {key: target_share[key] / source_share[key] for key in target_share}
    for cap in sorted({1.0, 1.5, 2.0, float(trim_cap)}):
        capped = _cap_weights(raw, target_share, source_share, cap)
        result = _weighted_estimate(source_table, capped)
        scenarios.append({
            'id': 'weight_cap_' + _number(cap),
            'type': 'weighting',
            'assumption': '单格目标/来源权重比上限 ' + _number(cap),
            'estimate': result['estimate'],
            'max_cell_weight': result['max_cell_weight'],
            'active': any(abs(capped[key] - target_share[key]) > 1e-12 for key in target_share),
        })

    # Deliberately wrong measurement assumption: treat `network` as comparable
    # because both sides spell the levels the same way.  Kept only to size the
    # error; it is never a reported conclusion.
    naive = _measurement_mismatch_scenario(source, target, columns)
    naive['difference_from_primary'] = (
        naive['estimate'] - primary_estimate
        if naive['estimate'] is not None and primary_estimate is not None else None)
    scenarios.append(naive)

    source_missing = {column: sum(1 for row in source if row.get(column) is None)
                      for column in transport_columns}
    target_missing = {column: sum(1 for row in target if row.get(column) is None)
                      for column in transport_columns}
    missing_weight = sum(record['target_weight'] for record in coverage['cells']
                         if record.get('reason') == 'missing_covariate')
    if missing_weight:
        total = coverage['total_target_weight']
        primary_share = coverage['supported_target_weight'] / total if total else 0.0
        interval = [primary_share * 0.0, primary_share * 1.0 + missing_weight / total]
        note = '缺失权重按完全未知处理：区间宽于把缺失格补零或直接丢弃的做法。'
    else:
        interval = None
        note = '本次没有目标缺失协变量；若出现，缺失权重会直接缩小可支持区域。'
    scenarios.append({
        'id': 'missing_covariate',
        'type': 'missingness',
        'assumption': '缺失水平单独成格，无法判定则归入 unsupported，不补零',
        'estimate': None,
        'interval': interval,
        'source_missing_rows': source_missing,
        'target_missing_rows': target_missing,
        'target_weight_with_missing_covariate': missing_weight,
        'note': note,
    })

    lab_outcomes = [1.0 if row['completed'] else 0.0 for row in source
                    if row['completed'] is not None]
    scenarios.append({
        'id': 'naive_lab_pooled',
        'type': 'no_transport',
        'assumption': '错误假设：忽略目标构成，直接用实验室总体完成率',
        'estimate': float(np.mean(lab_outcomes)) if lab_outcomes else None,
        'note': '这是不做迁移的参照，不应写成现网成绩。',
    })

    return {
        'scenarios': scenarios,
        'source_target_weight_ratio': {cell_label(key): value for key, value in raw.items()},
        'effective_sample_size': {
            'source_rows_in_support': int(sum(source_table[tuple(record['key'])]['rows']
                                              for record in supported)),
            'effective_rows_after_weighting':
                _effective_rows(target_share, source_table),
            'note': '有效样本按目标权重的集中度计算；明显小于来源行数时由少数格子主导。',
        },
        'uncheckable': [
            '未招募硬件层（例如 entry）的完成率：没有数据，只能给全人群边界',
            '测量含义不同的字段：即使同名编码也不能交换',
            '现场复测的选择机制：selected_probability 缺失时只能做情景分析',
        ],
    }


# ---------------------------------------------------------------------------
# validation


def _validation(validation, transport_columns, source_table, z, holdout_only=True):
    if not validation:
        return {'status': 'missing', 'reason': '没有现场复测记录',
                'used_for_fit': False}
    rows = []
    for index, row in enumerate(validation):
        if not isinstance(row, dict):
            raise ValueError('validation 第 ' + str(index) + ' 行不是对象')
        if 'completed' not in row:
            raise ValueError('validation 行 ' + str(row.get('row_id', index)) + ' 缺少 completed')
        if row.get('completed') is not None and not isinstance(row['completed'], bool):
            raise ValueError('validation 行 completed 必须是布尔值或 null')
        split = row.get('split')
        if split not in ('development', 'holdout', None):
            raise ValueError('validation split 必须是 development 或 holdout')
        rows.append(row)
    development = [row for row in rows if row.get('split') == 'development']
    holdout = [row for row in rows if row.get('split') == 'holdout']
    if not holdout:
        return {'status': 'no_holdout', 'reason': '没有 holdout 行，不能独立检验',
                'development_rows': len(development), 'used_for_fit': False}

    cells = {}
    for row in holdout:
        key = tuple(row.get(column) for column in transport_columns)
        cell = cells.setdefault(key, {'rows': 0, 'successes': 0, 'known_rows': 0,
                                      'known_successes': 0, 'unknown_rows': 0,
                                      'unknown_successes': 0,
                                      'probability_weight': 0.0,
                                      'probability_success': 0.0})
        completed = bool(row.get('completed'))
        cell['rows'] += 1
        cell['successes'] += 1 if completed else 0
        probability = row.get('selected_probability')
        if probability is None:
            cell['unknown_rows'] += 1
            cell['unknown_successes'] += 1 if completed else 0
        else:
            finite_weight(probability, 'selected_probability')
            if probability <= 0:
                raise ValueError('selected_probability 必须为正')
            cell['known_rows'] += 1
            cell['known_successes'] += 1 if completed else 0
            cell['probability_weight'] += 1.0 / probability
            cell['probability_success'] += (1.0 if completed else 0.0) / probability
    per_cell = []
    transport_gaps = []
    for key, cell in sorted(cells.items(), key=lambda item: tuple(str(v) for v in item[0])):
        observed = cell['successes'] / cell['rows']
        supported = key in source_table and source_table[key]['rate'] is not None
        expected = source_table[key]['rate'] if supported else None
        ipw = (cell['probability_success'] / cell['probability_weight']
               if cell['probability_weight'] else None)
        unknown_rate = (cell['unknown_successes'] / cell['unknown_rows']
                        if cell['unknown_rows'] else None)
        known_rate = (cell['known_successes'] / cell['known_rows']
                      if cell['known_rows'] else None)
        gap = observed - expected if expected is not None else None
        if supported and gap is not None:
            transport_gaps.append((cell['rows'], gap))
        per_cell.append({
            'key': list(key),
            'support': 'supported' if supported else 'unsupported',
            'rows': cell['rows'],
            'observed_successes': cell['successes'],
            'observed_rate': observed,
            'wilson_interval': _wilson(cell['successes'], cell['rows'], z),
            'source_supported_rate': expected,
            'transport_gap': gap,
            'selection_probability_known_rows': cell['known_rows'],
            'ipw_rate_where_known': ipw,
            'selection_bias_where_known': (ipw - known_rate
                                           if ipw is not None and known_rate is not None
                                           else None),
            'known_probability_rate': known_rate,
            'unknown_probability_rate': unknown_rate,
            'selection_mechanism_gap': (known_rate - unknown_rate
                                        if known_rate is not None and
                                        unknown_rate is not None else None),
        })

    total_rows = len(holdout)
    total_successes = sum(cell['successes'] for cell in cells.values())
    weighted_gap = (sum(rows * gap for rows, gap in transport_gaps) /
                    sum(rows for rows, _ in transport_gaps)) if transport_gaps else None
    return {
        'status': 'computed',
        'used_for_fit': False,
        'holdout_only': bool(holdout_only),
        'development_rows_excluded_from_validation': len(development),
        'holdout_rows': total_rows,
        'holdout_observed_rate': total_successes / total_rows if total_rows else None,
        'holdout_wilson_interval': _wilson(total_successes, total_rows, z),
        'per_cell': per_cell,
        'transport_error': {
            'mean_gap_holdout_minus_source_supported': weighted_gap,
            'cells_compared': len(transport_gaps),
            'note': '迁移误差只在与来源支持格子重合的复测格子上计算。',
        },
        'selection_bias': {
            'note': '选择偏差与迁移误差分开：前者在格子内用选择概率校正，'
                    '后者是复测率与来源支持格率之差。',
            'cells_with_known_probability': sum(1 for cell in cells.values()
                                                if cell['known_rows']),
            'selection_mechanism_gap_note': 'known_probability_rate 与 '
                                            'unknown_probability_rate 之差只能提示选择与结局相关，'
                                            '不能证明无偏。',
        },
        'random_truth': False,
        'note': '现场复测按业务挑选，不是随机真值；holdout 只用于独立检验，不参与拟合。',
    }


# ---------------------------------------------------------------------------
# fit


def fit(document, min_source_cell=DEFAULT_MIN_SOURCE_CELL, trim_cap=DEFAULT_TRIM_CAP,
        z=DEFAULT_Z):
    columns = _columns(document)
    source = _rows(document, 'source', require_outcome=True, weight_field=None)
    target = _rows(document, 'target', require_outcome=False, weight_field='population_weight')
    recruitment = _recruitment(document)

    measurement = assess_measurement(document)
    excluded = [column for column, info in measurement.items()
                if isinstance(info, dict) and not info.get('transportable', True)]
    transport_columns = [column for column in columns if column not in excluded]
    if not transport_columns:
        raise ValueError('没有任何测量含义可核对的迁移协变量')

    min_source_cell = int(min_source_cell)
    if min_source_cell < 1:
        raise ValueError('min_source_cell 必须为正整数')
    trim_cap = float(trim_cap)
    if not math.isfinite(trim_cap) or trim_cap < 1:
        raise ValueError('trim_cap 必须是不小于 1 的有限数')
    z = float(z)
    if not math.isfinite(z) or z <= 0:
        raise ValueError('z 必须为正的有限数')

    levels = levels_from(source, transport_columns)
    source_table = build_cell_table(source, transport_columns, outcome_field='completed')
    target_table = build_cell_table(target, transport_columns, weight_field='population_weight')
    records = classify_target_cells(target_table, transport_columns, source_table,
                                    min_source_cell, levels, recruitment)
    coverage = summarize_coverage(records)
    supported = [record for record in records if record['status'] == 'supported']
    if coverage['total_target_weight'] <= 0:
        raise ValueError('目标人群权重总和必须为正')
    if not supported:
        raise ValueError('没有目标格子获得来源支持，无法迁移；请检查招募限制与测量版本')

    target_share, source_share = _share_tables(source_table, target_table, supported)
    raw = {key: target_share[key] / source_share[key] for key in target_share}
    capped_share = _cap_weights(raw, target_share, source_share, trim_cap)

    primary = _weighted_estimate(source_table, target_share)
    trimmed = _weighted_estimate(source_table, capped_share)
    se_primary = _cluster_se(source_table, target_share)
    se_trimmed = _cluster_se(source_table, capped_share)

    lab_outcomes = [1.0 if row['completed'] else 0.0 for row in source
                    if row['completed'] is not None]
    lab_rate = float(np.mean(lab_outcomes)) if lab_outcomes else None

    model_estimate, extrapolation = _outcome_model_scenario(
        source, target, supported, transport_columns, target_share, z)

    estimates = [
        {
            'id': 'primary_transport',
            'label': '支持区域的目标加权完成率',
            'assumption': '支持格子内可交换：来源同格完成率代表目标同格完成率',
            'method': 'cell means weighted by target population share; no outcome model',
            'estimate': primary['estimate'],
            'se': se_primary,
            'interval': _normal_interval(primary['estimate'], se_primary, z),
            'population': 'supported_region_only',
            'supported_share': coverage['supported_share'],
            'max_cell_weight': primary['max_cell_weight'],
            'trimmed': False,
        },
        {
            'id': 'trimmed_transport',
            'label': '裁剪极端权重后的迁移完成率',
            'assumption': '同 primary，但单格目标/来源权重比不超过 ' + _number(trim_cap),
            'method': 'target shares with weight ratio capped at ' + _number(trim_cap),
            'estimate': trimmed['estimate'],
            'se': se_trimmed,
            'interval': _normal_interval(trimmed['estimate'], se_trimmed, z),
            'population': 'supported_region_only',
            'supported_share': coverage['supported_share'],
            'max_cell_weight': trimmed['max_cell_weight'],
            'trimmed': any(abs(capped_share[key] - target_share[key]) > 1e-12
                           for key in target_share),
        },
        {
            'id': 'lab_reference',
            'label': '不迁移的实验室总体参照',
            'assumption': '无迁移：直接把实验室样本当作现网',
            'method': 'lab row composition',
            'estimate': lab_rate,
            'se': None,
            'interval': _wilson(sum(lab_outcomes), len(lab_outcomes), z) if lab_outcomes else None,
            'population': 'lab_sample',
            'supported_share': None,
            'max_cell_weight': None,
            'trimmed': False,
        },
        {
            'id': 'outcome_model_scenario',
            'label': '可加 logistic 结果模型（仅支持区域）',
            'assumption': '完成率对迁移协变量 logit 可加；作为与 primary 实质不同的方案',
            'method': 'source-only logit averaged over supported target rows',
            'estimate': model_estimate['estimate'],
            'se': model_estimate['se'],
            'interval': model_estimate.get('interval'),
            'population': 'supported_region_only',
            'supported_share': coverage['supported_share'],
            'max_cell_weight': None,
            'trimmed': False,
        },
        {
            'id': 'whole_population_bounds',
            'label': '全人群上下界（含不支持人群）',
            'assumption': '不支持人群的完成率只假设落在 [0,1]，不给点估计',
            'method': 'supported estimate scaled by supported share',
            'estimate': None,
            'se': None,
            'interval': _population_bounds(primary['estimate'], coverage),
            'population': 'whole_target',
            'supported_share': coverage['supported_share'],
            'max_cell_weight': None,
            'trimmed': False,
        },
    ]

    sensitivity = _sensitivity(source, target, supported, columns,
                               transport_columns, target_share, source_share, source_table,
                               coverage, trim_cap, primary['estimate'])
    # The scenario already reports its own difference from primary; ensure it is
    # filled even if the scenario helper ran without knowing the primary value.
    for scenario in sensitivity['scenarios']:
        if scenario['id'] == 'naive_meaning_mismatch':
            scenario['difference_from_primary'] = (
                scenario['estimate'] - primary['estimate']
                if scenario['estimate'] is not None and primary['estimate'] is not None
                else None)
    sensitivity['extrapolation'] = extrapolation

    validation = _validation(document.get('validation') or [], transport_columns,
                             source_table, z)

    artifact = {
        'format': ARTIFACT_FORMAT,
        'version': ARTIFACT_VERSION,
        'fingerprint_source': _fingerprint(document, columns),
        'columns': columns,
        'transport_columns': transport_columns,
        'excluded_columns': excluded,
        'measurement_versions': document.get('measurement_versions') or [],
        'measured_versions_by_column': {
            info['field']: {'source': info.get('source_meanings', {}),
                            'target': info.get('target_meanings', {}),
                            'status': info.get('status'),
                            'source_versions': info.get('source_versions', []),
                            'target_versions': info.get('target_versions', [])}
            for info in measurement.values() if isinstance(info, dict)
        },
        'row_measurement_versions': {
            'source': sorted({str(row.get('measurement_version')) for row in source
                              if row.get('measurement_version') is not None}),
            'target': sorted({str(row.get('measurement_version')) for row in target
                              if row.get('measurement_version') is not None}),
        },
        'recruitment': recruitment,
        'levels': {column: [str(level) for level in values]
                   for column, values in levels.items()},
        'support_rule': {
            'min_source_cell': min_source_cell,
            'excluded_tiers': recruitment['excluded_tiers'],
            'required_network': recruitment['required_network'],
            'levels_seen_in_source': {column: [str(level) for level in values]
                                      for column, values in levels.items()},
        },
        'source_cells': {
            cell_label(key): {'rows': item['rows'], 'outcomes': item['outcomes'],
                              'successes': item['successes'], 'rate': item['rate']}
            for key, item in source_table.items()
        },
        'supported_cells': [{'key': list(record['key']),
                             'source_rows': source_table[tuple(record['key'])]['rows'],
                             'target_weight_at_fit': record['target_weight']}
                            for record in supported],
        'target_share': {cell_label(key): value for key, value in target_share.items()},
        'source_share': {cell_label(key): value for key, value in source_share.items()},
        'primary_estimate': primary['estimate'],
        'trim_cap': trim_cap,
        'z': z,
        'min_source_cell': min_source_cell,
        'source_rows': len(source),
        'target_rows': len(target),
    }

    result = {
        'target_coverage': {
            'columns_used': transport_columns,
            'excluded_columns': excluded,
            'supported_share': coverage['supported_share'],
            'unsupported_share': coverage['unsupported_share'],
            'supported_target_weight': coverage['supported_target_weight'],
            'total_target_weight': coverage['total_target_weight'],
            'unsupported_weight_by_reason': coverage['unsupported_weight_by_reason'],
            'cells': coverage['cells'],
            'unsupported_cells': coverage['unsupported_cells'],
            'whole_population_point_estimate': '拒绝',
            'claim_boundary': ('只能对 supported 区域给出估计；unsupported 区域只进入全人群边界。'
                               if coverage['supported_share'] < 1 else
                               '所有目标格均有来源支持，但测量含义与可交换性仍需核对。'),
        },
        'estimates': estimates,
        'balance': {
            'cells': _balance_cells(source_table, target_table, supported, target_share,
                                    source_share, raw),
            'marginals': _marginals(source, target, columns, transport_columns),
            'effective_source_rows_after_weighting': _effective_rows(target_share, source_table),
            'max_weight_ratio': max(raw.values()) if raw else None,
            'min_weight_ratio': min(raw.values()) if raw else None,
            'note': '权重比 = 目标格份额 / 来源格份额；只在支持区域内归一化。'
                    '排除字段也列出边际，但不能用于校正权重。',
        },
        'sensitivity': sensitivity,
        'validation': validation,
        'assumptions': {
            'checkable': [
                '支持判定：来源是否覆盖目标格子、来源样本量是否足够',
                '测量含义：measurement_versions 中同名版本的含义是否一致',
                '边际平衡：目标份额与来源份额的权重比、有效样本',
                '结果模型：仅用来源数据拟合，并在 holdout 上独立核对',
            ],
            'uncheckable_only_sensitivity': [
                '未覆盖硬件层的完成率：只能给全人群边界，不能给点估计',
                '同名编码但测量含义不同的字段：不能交换，只能演示偏差',
                '现场复测选择机制：selected_probability 缺失时只能做情景分析',
            ],
            'background_difference_not_adjustable': _background_difference(
                source, target, supported, columns, transport_columns, excluded,
                recruitment),
        },
    }
    return result, artifact


def _background_difference(source, target, supported, columns, transport_columns,
                           excluded, recruitment):
    """Describe differences on fields that cannot be used for weighting.

    These rows are not a bias adjustment.  They are the differences that remain
    inside the supported cells after the transportable fields are matched, so a
    reader can see why the exchangeability assumption is doing real work.
    """
    if not excluded:
        return {'status': 'none', 'fields': []}
    supported_keys = {tuple(record['key']) for record in supported}
    fields = []
    for column in excluded:
        source_rows = [row for row in source]
        target_rows = [row for row in target
                       if tuple(row.get(name) for name in transport_columns) in supported_keys]
        source_counts = {}
        for row in source_rows:
            source_counts[str(row.get(column))] = source_counts.get(str(row.get(column)), 0) + 1
        target_weight = {}
        for row in target_rows:
            target_weight[str(row.get(column))] = (target_weight.get(str(row.get(column)), 0.0)
                                                   + float(row['population_weight']))
        source_total = sum(source_counts.values())
        target_total = sum(target_weight.values())

        # Aggregated numbers can hide that one supported cell is matched on the
        # excluded field and another is not, so break it down per cell.
        per_cell = []
        for key in sorted(supported_keys, key=lambda value: tuple(str(v) for v in value)):
            cell_source = [row for row in source
                           if tuple(row.get(name) for name in transport_columns) == key]
            cell_target = [row for row in target
                           if tuple(row.get(name) for name in transport_columns) == key]
            if not cell_source or not cell_target:
                continue
            source_cell_counts = {}
            for row in cell_source:
                value = str(row.get(column))
                source_cell_counts[value] = source_cell_counts.get(value, 0) + 1
            target_cell_weight = {}
            for row in cell_target:
                value = str(row.get(column))
                target_cell_weight[value] = (target_cell_weight.get(value, 0.0)
                                             + float(row['population_weight']))
            source_cell_total = sum(source_cell_counts.values())
            target_cell_total = sum(target_cell_weight.values())
            per_cell.append({
                'key': list(key),
                'source_share': {value: count / source_cell_total
                                 for value, count in sorted(source_cell_counts.items())},
                'target_share': {value: weight / target_cell_total
                                 for value, weight in sorted(target_cell_weight.items())},
            })

        fields.append({
            'field': column,
            'why_excluded': '测量含义不可比或版本无法核对',
            'source_share_within_supported': {
                key: value / source_total for key, value in sorted(source_counts.items())}
            if source_total else {},
            'target_share_within_supported': {
                key: value / target_total for key, value in sorted(target_weight.items())}
            if target_total else {},
            'per_supported_cell': per_cell,
            'note': '这些差异不能被权重校正；若它与完成率相关，支持的格子内仍存在不可检查的混杂。'
                    '按格拆开可以看哪些格子内差异更大。',
        })
    return {'status': 'present', 'fields': fields}


def _population_bounds(supported_estimate, coverage):
    if supported_estimate is None:
        return None
    share = coverage['supported_share'] or 0.0
    return [share * supported_estimate, share * supported_estimate + (1 - share)]


def _balance_cells(source_table, target_table, supported, target_share, source_share, raw):
    cells = []
    for record in supported:
        key = tuple(record['key'])
        cells.append({
            'key': list(key),
            'source_rows': source_table[key]['rows'],
            'source_devices': source_table[key]['device_count'],
            'target_rows': target_table[key]['rows'],
            'target_weight': target_table[key]['weight'],
            'target_share': target_share[key],
            'source_share': source_share[key],
            'weight_ratio': raw[key],
            'source_rate': source_table[key]['rate'],
        })
    return cells


# ---------------------------------------------------------------------------
# apply


def apply(artifact, document):
    if not isinstance(artifact, dict) or artifact.get('format') != ARTIFACT_FORMAT:
        raise ValueError('不是 transport 产物')
    if artifact.get('version') != ARTIFACT_VERSION:
        raise ValueError('transport 产物版本不受支持')
    columns = _columns(document)
    if list(columns) != list(artifact.get('columns', [])):
        raise ValueError('新快照的 columns 与产物不一致，不能直接 apply')
    target = _rows(document, 'target', require_outcome=False, weight_field='population_weight')
    ignored = {
        'target_rows_with_completed_field': sum(1 for row in target if 'completed' in row),
        'target_outcomes_read': False,
        'note': 'apply 只读新快照的上线前协变量与人口权重；'
                '若新快照带了结局字段，它们被忽略且不会改变产物估计。',
    }
    transport_columns = artifact['transport_columns']
    supported_cells = {tuple(item['key']): item for item in artifact['supported_cells']}
    source_outcomes = artifact['source_cells']

    measured = artifact.get('measured_versions_by_column') or {}
    metadata = {(str(entry.get('field')), str(entry.get('version'))): str(entry.get('meaning'))
                for entry in artifact.get('measurement_versions') or []}
    for entry in document.get('measurement_versions') or []:
        if not isinstance(entry, dict) or not all(name in entry for name in
                                                  ('field', 'version', 'meaning')):
            raise ValueError('新快照的 measurement_versions 条目不完整')
        metadata[(str(entry['field']), str(entry['version']))] = str(entry['meaning'])
    version_warnings = []
    unchecked_columns = []
    excluded_drift = []
    transport_columns_verified = []
    for column in columns:
        if column not in transport_columns:
            excluded_drift.append({
                'column': column,
                'fit_status': (measured.get(column) or {}).get('status'),
                'fit_reason': (measured.get(column) or {}).get('reason'),
                'note': '该字段不参与迁移，新快照版本变化不会改变 applied 估计，'
                        '但会改变可迁移范围。',
            })
            continue
        info = measured.get(column) or {}
        if not info or info.get('status') == 'unversioned':
            # No meaning metadata: the only check possible is that the new
            # snapshot still carries row-level versions at all.  A changed set
            # of version labels is not proof of a meaning change, but it is a
            # reason to stop and re-fit rather than silently reuse the artifact.
            unchecked_columns.append(column)
            new_versions_now = {str(row.get('measurement_version')) for row in target
                                if row.get('measurement_version') is not None}
            fit_versions = set(artifact.get('row_measurement_versions', {}).get('target', []))
            if not new_versions_now or (fit_versions and
                                        not new_versions_now.issubset(fit_versions)):
                version_warnings.append({
                    'column': column, 'status': 'version_drift',
                    'reason': ('新快照行级版本缺失或出现拟合时未记录的行级版本' if
                               fit_versions else '新快照行级版本缺失')})
            continue
        fit_target_meanings = set((info.get('target') or {}).values())
        new_versions = {row.get('measurement_version') for row in target
                        if row.get('measurement_version') is not None}
        if not new_versions:
            version_warnings.append({
                'column': column, 'status': 'missing_row_version',
                'reason': '新快照行级 measurement_version 缺失'})
            continue
        missing_entry = sorted(str(version) for version in new_versions
                               if (column, str(version)) not in metadata)
        if missing_entry:
            version_warnings.append({
                'column': column, 'status': 'incomplete_metadata',
                'reason': '新快照测量版本没有含义记录：' + ', '.join(missing_entry)})
            continue
        new_meanings = {metadata[(column, str(version))] for version in new_versions}
        if len(new_meanings) != 1 or new_meanings != fit_target_meanings:
            version_warnings.append({
                'column': column, 'status': 'meaning_mismatch',
                'reason': '新快照测量含义与拟合时的目标含义不同'})
        else:
            transport_columns_verified.append(column)
    if version_warnings:
        raise ValueError('新快照的测量版本与产物不一致：' +
                         ', '.join(item['column'] + '(' + str(item['status']) + ')'
                                   for item in version_warnings))

    target_table = build_cell_table(target, transport_columns, weight_field='population_weight')
    support_rule = artifact.get('support_rule') or {}
    excluded_tiers = {str(value) for value in support_rule.get('excluded_tiers', [])}
    required_network = support_rule.get('required_network')
    seen_levels = support_rule.get('levels_seen_in_source') or {}
    min_rows = int(support_rule.get('min_source_cell', 1))
    source_cells = source_outcomes
    records = []
    for key in sorted(target_table, key=lambda value: tuple(str(v) for v in value)):
        item = target_table[key]
        cell = source_cells.get(cell_label(key))
        tier = key[transport_columns.index('tier')] if 'tier' in transport_columns else None
        network = key[transport_columns.index('network')] \
            if 'network' in transport_columns else None
        unseen = [column for column, value in zip(transport_columns, key)
                  if str(value) not in {str(level) for level in seen_levels.get(column, [])}]
        if any(value is None for value in key):
            status, reason = 'unsupported', 'missing_covariate'
        elif tier is not None and str(tier) in excluded_tiers:
            status, reason = 'unsupported', 'recruitment_excluded'
        elif required_network is not None and network is not None and \
                str(network) != str(required_network):
            status, reason = 'unsupported', 'recruitment_restriction'
        elif unseen:
            status, reason = 'unsupported', 'unseen_level'
        elif cell is None:
            status, reason = 'unsupported', 'no_source_support'
        elif cell['rows'] < min_rows:
            status, reason = 'unsupported', 'insufficient_source_support'
        else:
            status, reason = 'supported', None
        records.append({'key': list(key), 'target_weight': item['weight'],
                        'target_rows': item['rows'], 'status': status, 'reason': reason})
    total = sum(record['target_weight'] for record in records)
    supported = [record for record in records if record['status'] == 'supported']
    supported_total = sum(record['target_weight'] for record in supported)
    weights = {tuple(record['key']): record['target_weight'] / supported_total
               for record in supported} if supported_total else {}

    estimate = None
    if weights:
        estimate = 0.0
        for key, weight in weights.items():
            cell = source_outcomes[cell_label(key)]
            if not cell['outcomes']:
                raise ValueError('产物中的支持格子没有可用结局')
            estimate += weight * (cell['successes'] / cell['outcomes'])

    balance = []
    fit_shares = artifact.get('target_share') or {}
    for record in supported:
        key = tuple(record['key'])
        fit_share = fit_shares.get(cell_label(key))
        new_share = record['target_weight'] / supported_total
        balance.append({
            'key': list(key),
            'fit_target_share': fit_share,
            'apply_target_share': new_share,
            'share_drift': (new_share - fit_share) if fit_share is not None else None,
            'source_rate': source_outcomes[cell_label(key)]['rate'],
            'seen_in_fit_target': fit_share is not None,
        })
    return {
        'target_coverage': {
            'columns_used': transport_columns,
            'supported_share': supported_total / total if total else None,
            'unsupported_share': 1 - supported_total / total if total else None,
            'supported_target_weight': supported_total,
            'total_target_weight': total,
            'cells': records,
        },
        'estimates': [{
            'id': 'applied_transport',
            'label': '把拟合时的支持格率应用到新快照',
            'assumption': '沿用产物中的可交换性假设与支持判定',
            'estimate': estimate,
            'population': 'supported_region_only',
            'supported_share': supported_total / total if total else None,
            'artifact_version': artifact['version'],
            'source_fingerprint': artifact['fingerprint_source'],
        }],
        'balance': {
            'cells': balance,
            'note': '新快照的支持格份额与拟合时份额对比；漂移大时估计应重做支持判定。',
        },
        'sensitivity': {
            'status': 'not_recomputed',
            'reason': 'apply 不读取任何目标结局，也不重新拟合；敏感性结论沿用产物。',
        },
        'validation': {
            'status': 'not_run',
            'reason': 'apply 的新快照没有结局，不能用现场复测重新检验。',
        },
        'measurement_check': {
            'transport_columns_verified': transport_columns_verified,
            'unchecked_unversioned_columns': unchecked_columns,
            'excluded_column_drift': excluded_drift,
            'note': '迁移字段的含义必须与拟合时一致；unversioned 字段无法核对，只能假设一致。',
        },
        'artifact': {
            'source_fingerprint': artifact['fingerprint_source'],
            'source_rows': artifact.get('source_rows'),
            'supported_cells_at_fit': len(supported_cells),
            'trim_cap': artifact.get('trim_cap'),
            'recomputed_from_target_outcomes': False,
            'ignored_inputs': ignored,
            'note': '只使用产物内来源格率与本次快照的上线前协变量。',
        },
    }



# ---------------------------------------------------------------------------
# module entry point: python3 -m field_transport_lab.transport
#
# `__main__.main` owns the shared parser but is written for the top-level
# command (`inspect`/`transport`).  The transport module is itself a command,
# so it prepends its own subcommand name and reuses the same implementation.


def _main(argv=None):
    from .__main__ import main as package_main
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in ('fit', 'apply'):
        arguments = ['transport'] + arguments
    return package_main(arguments)


if __name__ == '__main__':
    sys.exit(_main())
