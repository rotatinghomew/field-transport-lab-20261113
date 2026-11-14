"""Support, measurement-compatibility and cell utilities for transport analysis.

The functions here only look at pre-treatment covariates, recruitment metadata
and measurement-version metadata.  Outcomes are never read.
"""

import math
from collections import defaultdict


def cell_key(row, columns):
    return tuple(row.get(column) for column in columns)


def cell_label(key):
    return '|'.join('' if value is None else str(value) for value in key)


def levels_from(rows, columns):
    return {
        column: sorted({row.get(column) for row in rows if row.get(column) is not None},
                       key=str)
        for column in columns
    }


def build_cell_table(rows, columns, weight_field=None, outcome_field=None):
    """Aggregate rows by covariate cell.

    ``weight`` is the sum of population weights (1 for source rows).
    ``outcomes``/``successes`` are only populated when ``outcome_field`` is given;
    rows whose outcome is None count as not observed and never as failure.
    """
    table = {}
    for row in rows:
        key = cell_key(row, columns)
        item = table.get(key)
        if item is None:
            item = {'key': list(key), 'rows': 0, 'weight': 0.0,
                    'outcomes': 0, 'successes': 0.0, 'weight_observed': 0.0,
                    'devices': set(), 'device_stats': {}, 'missing_covariate': False}
            table[key] = item
        weight = row.get(weight_field, 1) if weight_field else 1.0
        item['rows'] += 1
        item['weight'] += weight
        device = row.get('device_id')
        if device is not None:
            item['devices'].add(str(device))
            stats = item['device_stats'].setdefault(str(device), {'rows': 0, 'successes': 0.0})
            stats['rows'] += 1
            if outcome_field is not None and row.get(outcome_field) is not None:
                stats['successes'] += 1.0 if row.get(outcome_field) else 0.0
        if any(row.get(column) is None for column in columns):
            item['missing_covariate'] = True
        if outcome_field is not None:
            outcome = row.get(outcome_field)
            if outcome is not None:
                item['outcomes'] += 1
                item['weight_observed'] += weight
                item['successes'] += weight * (1.0 if outcome else 0.0)
    for item in table.values():
        item['rate'] = (item['successes'] / item['weight_observed']
                        if item['weight_observed'] > 0 else None)
        item['device_count'] = len(item['devices'])
    return table


def assess_measurement(document):
    """Compare measurement metadata between source and target.

    A field is transportable only when its meaning can be established as the
    same on both sides.  Same JSON encoding (``normal``/``weak``) is not enough.
    Fields without any version metadata stay usable, but the assumption is
    recorded as unverifiable rather than silently treated as verified.
    """
    columns = document['columns']
    source = document['source']
    target = document['target']
    entries = defaultdict(dict)
    extra = []
    for entry in document.get('measurement_versions') or []:
        if not isinstance(entry, dict):
            raise ValueError('measurement_versions 条目必须是对象')
        for name in ('field', 'version', 'meaning'):
            if name not in entry:
                raise ValueError('measurement_versions 条目缺少 ' + name)
        if entry['field'] not in columns:
            extra.append(entry['field'])
            continue
        entries[entry['field']][str(entry['version'])] = str(entry['meaning'])

    result = {}
    for column in columns:
        source_versions = sorted({row.get('measurement_version') for row in source
                                  if row.get('measurement_version') is not None}, key=str)
        target_versions = sorted({row.get('measurement_version') for row in target
                                  if row.get('measurement_version') is not None}, key=str)
        # A row that silently omits its version cannot be checked; treating the
        # omission as "same as the others" would hide a version change.
        source_missing = sum(1 for row in source if row.get('measurement_version') is None)
        target_missing = sum(1 for row in target if row.get('measurement_version') is None)
        field_entries = entries.get(column)
        info = {
            'field': column,
            'source_versions': source_versions,
            'target_versions': target_versions,
            'source_missing_version_rows': source_missing,
            'target_missing_version_rows': target_missing,
            'source_meanings': {},
            'target_meanings': {},
        }
        if not field_entries:
            info['status'] = 'unversioned'
            info['transportable'] = True
            info['reason'] = '没有测量版本元数据；只能假设两边编码含义一致'
        else:
            source_meanings = {version: field_entries.get(version) for version in source_versions}
            target_meanings = {version: field_entries.get(version) for version in target_versions}
            info['source_meanings'] = {k: v for k, v in source_meanings.items() if v is not None}
            info['target_meanings'] = {k: v for k, v in target_meanings.items() if v is not None}
            if not source_versions or not target_versions:
                info['status'] = 'missing_row_version'
                info['transportable'] = False
                info['reason'] = '行级 measurement_version 缺失，无法核对测量含义'
            elif source_missing or target_missing:
                info['status'] = 'missing_row_version'
                info['transportable'] = False
                info['reason'] = ('部分行缺少 measurement_version，无法确认这些行与已记录版本一致'
                                  if source_missing and target_missing else
                                  '有一侧的部分行缺少 measurement_version')
            elif any(value is None for value in source_meanings.values()) or \
                    any(value is None for value in target_meanings.values()):
                info['status'] = 'incomplete_metadata'
                info['transportable'] = False
                info['reason'] = '某个行级测量版本没有对应的含义记录'
            else:
                source_set = set(source_meanings.values())
                target_set = set(target_meanings.values())
                if len(source_set) == 1 and len(target_set) == 1 and source_set == target_set:
                    info['status'] = 'comparable'
                    info['transportable'] = True
                    info['reason'] = '两边测量含义一致'
                else:
                    info['status'] = 'meaning_mismatch'
                    info['transportable'] = False
                    info['reason'] = '两边编码相同但测量含义不同，不能作为迁移协变量'
        result[column] = info
    if extra:
        result['_fields_not_in_columns'] = sorted(set(extra))
    return result


def classify_target_cells(target_table, columns, source_table, min_rows,
                          levels, recruitment):
    """Return one record per target cell with support status and reason."""
    excluded_tiers = {str(value) for value in
                      (recruitment or {}).get('excluded_tiers', []) or []}
    required_network = (recruitment or {}).get('required_network')
    records = []
    for key in sorted(target_table, key=lambda k: tuple(str(v) for v in k)):
        item = target_table[key]
        missing = [column for column, value in zip(columns, key) if value is None]
        record = {
            'key': list(key),
            'target_weight': item['weight'],
            'target_rows': item['rows'],
            'source_rows': source_table.get(key, {}).get('rows', 0),
            'source_rate': source_table.get(key, {}).get('rate'),
            'source_devices': source_table.get(key, {}).get('device_count', 0),
        }
        if missing:
            status, reason = 'unsupported', 'missing_covariate'
            detail = '目标行缺少协变量：' + ', '.join(missing)
        elif 'tier' in columns and str(key[columns.index('tier')]) in excluded_tiers:
            status, reason = 'unsupported', 'recruitment_excluded'
            detail = '来源招募明确排除了该 tier'
        elif required_network is not None and 'network' in columns and \
                str(key[columns.index('network')]) != str(required_network):
            status, reason = 'unsupported', 'recruitment_restriction'
            detail = '来源招募只覆盖 required_network=' + str(required_network)
        else:
            unseen = [column + '=' + str(value) for column, value in zip(columns, key)
                      if value not in levels.get(column, [])]
            if unseen:
                status, reason = 'unsupported', 'unseen_level'
                detail = '来源没有出现过的水平：' + ', '.join(unseen)
            elif key not in source_table:
                status, reason = 'unsupported', 'no_source_support'
                detail = '来源没有该协变量组合'
            elif source_table[key]['rows'] < min_rows:
                status, reason = 'unsupported', 'insufficient_source_support'
                detail = '来源该格样本少于 ' + str(min_rows)
            else:
                status, reason = 'supported', None
                detail = '来源有足够同格样本'
        record['status'] = status
        record['reason'] = reason
        record['detail'] = detail
        records.append(record)
    return records


def summarize_coverage(records):
    total = sum(record['target_weight'] for record in records)
    supported = sum(record['target_weight'] for record in records
                    if record['status'] == 'supported')
    unsupported = []
    for record in records:
        if record['status'] == 'supported':
            continue
        unsupported.append({
            'key': record['key'],
            'target_weight': record['target_weight'],
            'share': record['target_weight'] / total if total else None,
            'reason': record['reason'],
            'detail': record['detail'],
        })
    reasons = defaultdict(float)
    for record in records:
        if record['status'] != 'supported':
            reasons[record['reason']] += record['target_weight']
    return {
        'total_target_weight': total,
        'supported_target_weight': supported,
        'supported_share': supported / total if total else None,
        'unsupported_share': (total - supported) / total if total else None,
        'cells': records,
        'unsupported_cells': unsupported,
        'unsupported_weight_by_reason': dict(sorted(reasons.items())),
    }


def finite_weight(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(value) or value < 0:
        raise ValueError(label + '无效')
    return float(value)
