import sys
# update your projecty root path before running
sys.path.insert(0, '/content/nsga-net')

import os
import ast
import time
import shutil
import logging
import argparse
from misc import utils

import numpy as np
from search import train_search
from search import micro_encoding
from search import macro_encoding
from search import nsganet as engine

from pymop.problem import Problem
from pymoo.optimize import minimize

parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for NAS")
parser.add_argument('--save', type=str, default='GA-BiObj', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--search_space', type=str, default='micro', help='macro or micro search space')
parser.add_argument('--resume', type=int, default=0, help='resume optimization')
# arguments for micro search space
parser.add_argument('--n_blocks', type=int, default=5, help='number of blocks in a cell')
parser.add_argument('--n_ops', type=int, default=9, help='number of operations considered')
parser.add_argument('--n_cells', type=int, default=2, help='number of cells to search')
# arguments for macro search space
parser.add_argument('--n_nodes', type=int, default=4, help='number of nodes per phases')
# hyper-parameters for algorithm
parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
parser.add_argument('--n_gens', type=int, default=50, help='population size')
parser.add_argument('--n_offspring', type=int, default=40, help='number of offspring created per generation')
# arguments for back-propagation training during search
parser.add_argument('--init_channels', type=int, default=24, help='# of filters for first cell')
parser.add_argument('--layers', type=int, default=11, help='equivalent with N = 3')
parser.add_argument('--epochs', type=int, default=25, help='# of epochs to train during architecture search')
args = parser.parse_args()
if not args.resume:
    args.save = 'search-{}-{}-{}'.format(args.save, args.search_space, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join("/content/drive/MyDrive", args.save)
    utils.create_exp_dir(args.save)
    utils.create_exp_dir(os.path.join(args.save, "populations"))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log4.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

pop_hist = []  # keep track of every evaluated architecture

# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, search_space='micro', n_var=20, n_obj=1, n_constr=0, lb=None, ub=None,
                 init_channels=24, layers=8, epochs=25, save_dir=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=int)
        self.xl = lb
        self.xu = ub
        self._search_space = search_space
        self._init_channels = init_channels
        self._layers = layers
        self._epochs = epochs
        self._save_dir = save_dir
        self._n_evaluated = 0 # keep track of how many architectures are sampled

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('\n')
            logging.info('Network id = {}'.format(arch_id))

            # call back-propagation training
            if self._search_space == 'micro':
                genome = micro_encoding.convert(x[i, :])
            elif self._search_space == 'macro':
                genome = macro_encoding.convert(x[i, :])

            if arch_id == 841:
                raise
            
            arch_file = os.path.join(self._save_dir, 'arch_{}'.format(arch_id), 'log.txt')
            need_train = True

            if os.path.isfile(arch_file):
                with open(arch_file, 'r') as f:
                    lines = f.readlines()

                log_genome = np.array(ast.literal_eval(lines[0].split(" = ")[1][:-1]))
                log_flops = float(lines[3].split(" = ")[1].split("MB")[0])
                log_valid_acc = float(lines[4].split(" = ")[1][:-1])
                log_params = float(lines[2].split(" = ")[1].split("MB")[0])

                if np.array_equal(genome, log_genome):

                    performance = {"flops": log_flops,
                                "valid_acc": log_valid_acc,
                                "params": log_params}

                    genotype = macro_encoding.decode(genome)
                    logging.info("Architecture = %s", genotype)
                    logging.info('valid_acc %f', performance['valid_acc'])
                    logging.info('flops = %f', performance['flops'])
                    need_train = False
                else:
                    logging.info("Logged genome not be equal to evaluated genome!")

            if need_train:
                if (arch_id % 40) >= 31 or (arch_id % 40) == 0:
                    performance = train_search.main(genome=genome,
                                                    search_space=self._search_space,
                                                    init_channels=self._init_channels,
                                                    layers=self._layers, cutout=False,
                                                    epochs=self._epochs,
                                                    save='arch_{}'.format(arch_id),
                                                    expr_root=self._save_dir)
                else:
                    performance = {"flops": 0.0,
                                "valid_acc": 0.0,
                                "params": 0.0}

            # all objectives assume to be MINIMIZED !!!!!
            objs[i, 0] = 100 - performance['valid_acc']
            objs[i, 1] = performance['flops']

            self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    algorithm.update_pop_archive(pop_var)

    # report generation info to files
    logging.info("generation = {}".format(gen))
    logging.info("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    logging.info("population complexity: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
                                                  np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))
    logging.info("population archive size: {}".format(len(algorithm.pop_archive)))

def main():
    np.random.seed(args.seed)
    logging.info("args = %s", args)

    # setup NAS search problem
    if args.search_space == 'micro':  # NASNet search space
        n_var = int(4 * args.n_blocks * 2)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
        h = 1
        for b in range(0, n_var//2, 4):
            ub[b] = args.n_ops - 1
            ub[b + 1] = h
            ub[b + 2] = args.n_ops - 1
            ub[b + 3] = h
            h += 1
        ub[n_var//2:] = ub[:n_var//2]
    elif args.search_space == 'macro':  # modified GeneticCNN search space
        n_var = int(((args.n_nodes-1)*args.n_nodes/2 + 1)*3)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
    else:
        raise NameError('Unknown search space type')

    problem = NAS(n_var=n_var, search_space=args.search_space,
                  n_obj=2, n_constr=0, lb=lb, ub=ub,
                  init_channels=args.init_channels, layers=args.layers,
                  epochs=args.epochs, save_dir=args.save)

    # configure the nsga-net method
    method = engine.nsganet(n_var=n_var,
                            pop_size=args.pop_size,
                            n_offsprings=args.n_offspring,
                            eliminate_duplicates=True)
    
    method.populations_dir = os.path.join(args.save, "populations")

    res = minimize(problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', args.n_gens))

    return


if __name__ == "__main__":
    main()