import json
from argparse import ArgumentParser
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules

if __name__ == '__main__':
    import_submodules('SMiLe_OIE')
    parser = ArgumentParser()
    parser.add_argument("--config", default='', help="input config json file")
    parser.add_argument("--model", default='', help="model output directory")
    parser.add_argument("--train_data", default='', help="training file")
    parser.add_argument("--eval_data", default='', help="evaluation file")
    parser.add_argument("--epoch", type=int, default=0, help="number of epoches")
    parser.add_argument("--batch", type=int, default=0, help="batch size")
    parser.add_argument("--cuda", type=int, default=0, help="batch size")
    parser.add_argument("--div", type=float, default=0.03, help="hyperparameter for multi-view div")
    parser.add_argument("--c1", type=float, default=0.02, help="hyperparameter for multi-view c1")
    parser.add_argument("--c2", type=float, default=0.02, help="hyperparameter for multi-view c2")
    args = parser.parse_args()

    overrideD = dict()
    overrideD['iterator'] = dict()
    overrideD['trainer'] = dict()
    overrideD['model'] = dict()
    overrideD['trainer']["cuda_device"] = args.cuda
    if args.div > 0:
        overrideD['model']['hyper_div'] = args.div
    if args.c1 > 0:
        overrideD['model']['hyper_c1'] = args.c1
    if args.c2 > 0:
        overrideD['model']['hyper_c2'] = args.c2
    if args.train_data != '' and args.eval_data != '':
        overrideD['train_data_path'] = args.train_data
        overrideD['validation_data_path'] = args.eval_data

    serialization_dir = args.model
    config_file = args.config
    serialization_dir = serialization_dir + "_{}_{}_{}".format(args.div, args.c1, args.c2)

    # writing predictions to output folders:
    overrideD['model']["tuple_metric"] = dict()
    overrideD['model']["tuple_metric"]["output_path"] = serialization_dir

    if args.epoch > 0:
        overrideD['trainer']["num_epochs"] = args.epoch
    if args.batch > 0:
        overrideD['iterator']["batch_size"] = args.batch

    overrides = json.dumps(overrideD)
    train_model_from_file(parameter_filename=config_file, serialization_dir=serialization_dir, recover=False, overrides=overrides)