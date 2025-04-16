from argparse import ArgumentParser
from pathlib import Path

import torch

from src.model.bert import config as bert_config
from src.model.modern_bert import config as modern_bert_config


def argparse():
    parser = ArgumentParser(description='Convert pl checkpoint to transformers format')
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--model_arch', type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


def main(args):
    ckpt_path = Path(args.ckpt_path)
    ckpt_dir = ckpt_path.parent

    ckpt = torch.load(ckpt_path, weights_only=False)
    state_dict = ckpt['state_dict']
    state_dict = {'.'.join(k.split('.')[2:]): v for k, v in state_dict.items()}

    # 同一ディレクトリにpytorch_model.binとconfig.jsonが必要
    state_dict_path = ckpt_dir / f'pytorch_model.bin'
    torch.save(state_dict, state_dict_path)
    if args.model_arch == 'modernbert':
        config = modern_bert_config
    else:
        config = bert_config
    config.to_json_file(ckpt_dir / 'config.json')


if __name__ == '__main__':
    args = argparse()
    main(args)
