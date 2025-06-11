import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.individual import Individual
from pymoo.model.survival import Survival
from pymoo.operators.crossover.point_crossover import PointCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.display import disp_multi_objective
from pymoo.util.dominator import Dominator
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort


# =========================================================================================================
# Implementation
# based on nsga2 from https://github.com/msu-coinlab/pymoo
# =========================================================================================================


class NSGANet(GeneticAlgorithm):

    def __init__(self, **kwargs):
        kwargs['individual'] = Individual(rank=np.inf, crowding=-1)
        super().__init__(**kwargs)

        self.tournament_type = 'comp_by_dom_and_crowding'
        self.func_display_attrs = disp_multi_objective
        self.pop_archive = []

    def update_pop_archive(self, pop_var):
        for i in range(pop_var.shape[0]):
            not_seen = True
            for p in self.pop_archive:
                if np.array_equal(pop_var[i], p):
                    not_seen = False
                    break
            if not_seen:
                self.pop_archive.append(pop_var[i])

    def _mating(self, pop):

        if self.n_gen <= 20:
            off = super()._mating(pop)
        
        else:
            off = self.heuristic_recombination(pop)
        
        return off
    
    def heuristic_recombination(self, pop):
        # the population object to be used
        off = pop.new()

        max_trail = 20

        # sampling counter - counts how often the sampling needs to be done to fill up n_offsprings
        n_sampling = 0

        # iterate until enough offsprings are created
        while len(off) < self.n_offsprings:
            trial = 1

            while True:
                conn = self.sample_conn_from_bayesian()
                duplicate = False  # assumes this conn is not duplicate
                # 1st check if conn exists in current population
                for i in range(pop.shape[0]):
                    if np.array_equal(conn, pop[i]):
                        duplicate = True
                        break
                # 2nd check if conn exists in current offspring
                for i in range(off.shape[0]):
                    if np.array_equal(conn, off[i]):
                        duplicate = True
                        break
                # 3rd check if conn exists in population archive
                for member in self.pop_archive:
                    if np.array_equal(conn, member):
                        duplicate = True
                        break
                if (not duplicate) or (trial > max_trail):
                    break
                # print(trial)
                trial += 1

            if not duplicate:
                _off = pop.new("X", np.reshape(conn, (1, -1)))

                # add to the offsprings
                off = off.merge(_off)

            # increase the mating counter
            n_sampling += 1

            # if no new offsprings can be generated within 100 trails -> return the current result
            if n_sampling > 100:
                print(
                    "WARNING: Recombination could not produce new offsprings which are not already in the population!")
                break
                

        return off

    def sample_conn_from_bayesian(self):
        conn = []

        len_phase = int(self.pop_archive.shape[1] / 3)

        for ph in range(3):
            if ph == 0:
                idx = np.random.randint(len(self.pop_archive))
                
            else:
                dependencies = []
                for i, member in enumerate(self.pop_archive):
                    phase = member[len_phase * (ph - 1) : len_phase * ph]
                    if np.array_equal(phase, conn[ph - 1]):
                        dependencies.append(i)
                idx = np.random.choice(dependencies)

            phase = self.pop_archive[idx][len_phase * ph : len_phase * (ph + 1)]
            conn.append(phase)

        conn = np.concatenate(conn)

        return conn

# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(pop[a].F, pop[b].F)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, pop[a].rank, b, pop[b].rank,
                               method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].get("crowding"), b, pop[b].get("crowding"),
                               method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(np.int)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(True)

    def _do(self, pop, n_survive, D=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F")

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


def calc_crowding_distance(F):
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # sort each column and get index
        I = np.argsort(F, axis=0, kind='mergesort')

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) \
               - np.concatenate([np.full((1, n_obj), -np.inf), F])

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity

    return crowding


# =========================================================================================================
# Interface
# =========================================================================================================


def nsganet(
        pop_size=100,
        sampling=RandomSampling(var_type=np.int),
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=PointCrossover(n_points=2),
        mutation=PolynomialMutation(eta=3, var_type=np.int),
        eliminate_duplicates=True,
        n_offsprings=None,
        **kwargs):
    """

    Parameters
    ----------
    pop_size : {pop_size}
    sampling : {sampling}
    selection : {selection}
    crossover : {crossover}
    mutation : {mutation}
    eliminate_duplicates : {eliminate_duplicates}
    n_offsprings : {n_offsprings}

    Returns
    -------
    nsganet : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an NSGANet algorithm object.


    """

    return NSGANet(pop_size=pop_size,
                   sampling=sampling,
                   selection=selection,
                   crossover=crossover,
                   mutation=mutation,
                   survival=RankAndCrowdingSurvival(),
                   eliminate_duplicates=eliminate_duplicates,
                   n_offsprings=n_offsprings,
                   **kwargs)


parse_doc_string(nsganet)
