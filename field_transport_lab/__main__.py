import argparse
import sys

from . import transport as transport_module
from .io import read_json, write_json
from .population import inspect


def _build_parser():
    parser = argparse.ArgumentParser(
        description='描述实验室与现网终端特征分布，保存基准任务成绩和目标人群快照。')
    subparsers = parser.add_subparsers(dest='command', required=True)

    inspect_parser = subparsers.add_parser('inspect', help='查看字段分布')
    inspect_parser.add_argument('--input', required=True)
    inspect_parser.add_argument('--output', required=True)

    transport_parser = subparsers.add_parser(
        'transport', help='从来源样本迁移到目标人群的完成率分析')
    transport_sub = transport_parser.add_subparsers(dest='subcommand', required=True)
    for name in ('fit', 'apply'):
        sub = transport_sub.add_parser(name)
        sub.add_argument('--input', required=True)
        sub.add_argument('--output', required=True)
        if name == 'apply':
            sub.add_argument('--artifact', required=True)
        else:
            sub.add_argument('--min-source-cell', type=int,
                             default=transport_module.DEFAULT_MIN_SOURCE_CELL)
            sub.add_argument('--trim-cap', type=float,
                             default=transport_module.DEFAULT_TRIM_CAP)
            sub.add_argument('--z', type=float, default=transport_module.DEFAULT_Z)
            sub.add_argument('--artifact', default=None,
                             help='产物路径；默认写到 <output>.artifact.json')
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        document = read_json(args.input)
        if args.command == 'inspect':
            result = inspect(document)
        elif args.subcommand == 'fit':
            result, artifact = transport_module.fit(
                document, min_source_cell=args.min_source_cell,
                trim_cap=args.trim_cap, z=args.z)
            artifact_path = args.artifact or (str(args.output) + '.artifact.json')
            write_json(artifact_path, artifact)
            result['artifact_path'] = artifact_path
        else:
            artifact = read_json(args.artifact)
            result = transport_module.apply(artifact, document)
        write_json(args.output, result)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print('处理失败：' + str(error), file=sys.stderr)
        return 2
    print('结果已保存：' + args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
