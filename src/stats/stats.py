from copy import copy
from sys import stdout
from time import time
from os import path

import numpy as np
import pandas as pd
from algorithm.parameters import params
from utilities.algorithm.NSGA2 import compute_pareto_metrics, compute_pareto_metrics_v2
from utilities.algorithm.state import create_state
from utilities.stats import trackers
from utilities.stats.file_io import save_best_ind_to_file, \
    save_first_front_to_file, save_stats_headers, save_stats_to_file, save_values_to_file
from utilities.stats.save_plots import save_pareto_fitness_plot, \
    save_plot_from_data
from utilities.fitness.error_metric import f1_score, accuracy
from utilities.representation.assoc_rules import RuleSet, AssocRules_Stats

"""Algorithm statistics"""
stats = {
    "gen": 0,
    "total_inds": 0,
    "regens": 0,
    "invalids": 0,
    "runtime_error": 0,
    "unique_inds": len(trackers.cache),
    "unused_search": 0,
    "ave_genome_length": 0,
    "max_genome_length": 0,
    "min_genome_length": 0,
    "ave_used_codons": 0,
    "max_used_codons": 0,
    "min_used_codons": 0,
    "ave_tree_depth": 0,
    "max_tree_depth": 0,
    "min_tree_depth": 0,
    "ave_tree_nodes": 0,
    "max_tree_nodes": 0,
    "min_tree_nodes": 0,
    "ave_fitness": 0,
    "best_fitness": 0,
    "time_taken": 0,
    "total_time": 0,
    "time_adjust": 0
}


def get_stats(individuals, end=False):
    """
    Generate the statistics for an evolutionary run. Save statistics to
    utilities.trackers.stats_list. Print statistics. Save fitness plot
    information.

    :param individuals: A population of individuals for which to generate
    statistics.
    :param end: Boolean flag for indicating the end of an evolutionary run.
    :return: Nothing.
    """

    if hasattr(params['FITNESS_FUNCTION'], 'multi_objective'):
        # Multiple objective optimisation is being used.

        # Remove fitness stats from the stats dictionary.
        stats.pop('best_fitness', None)
        stats.pop('ave_fitness', None)

        # Update stats.
        get_moo_stats_v2(individuals, end)

    else:
        # Single objective optimisation is being used.
        get_soo_stats_v2(individuals, end)

    if params['SAVE_STATE'] and not params['DEBUG'] and \
            stats['gen'] % params['SAVE_STATE_STEP'] == 0:
        # Save the state of the current evolutionary run.
        create_state(individuals)


def get_soo_stats(individuals, end):
    """
    Generate the statistics for an evolutionary run with a single objective.
    Save statistics to utilities.trackers.stats_list. Print statistics. Save
    fitness plot information.

    :param individuals: A population of individuals for which to generate
    statistics.
    :param end: Boolean flag for indicating the end of an evolutionary run.
    :return: Nothing.
    """

    # Get best individual.
    best = max(individuals)

    if not trackers.best_ever or best > trackers.best_ever:
        # Save best individual in trackers.best_ever.
        trackers.best_ever = best

    if end or params['VERBOSE'] or not params['DEBUG']:
        # Update all stats.
        update_stats(individuals, end)

    # Save fitness plot information
    if params['SAVE_PLOTS'] and not params['DEBUG']:
        if not end:
            trackers.best_fitness_list.append(trackers.best_ever.fitness)

        if params['VERBOSE'] or end:
            save_plot_from_data(trackers.best_fitness_list, "best_fitness")

    # Print statistics
    if params['VERBOSE'] and not end:
        print_generation_stats()

    elif not params['SILENT']:
        # Print simple display output.
        perc = stats['gen'] / (params['GENERATIONS'] + 1) * 100
        stdout.write("Evolution: %d%% complete\r" % perc)
        stdout.flush()

    # Generate test fitness on regression problems
    if hasattr(params['FITNESS_FUNCTION'], "training_test") and end:
        # Save training fitness.
        trackers.best_ever.training_fitness = copy(trackers.best_ever.fitness)

        # Evaluate test fitness.
        trackers.best_ever.test_fitness = params['FITNESS_FUNCTION'](
            trackers.best_ever, dist='test')

        # Set main fitness as training fitness.
        trackers.best_ever.fitness = trackers.best_ever.training_fitness

    # Save stats to list.
    if params['VERBOSE'] or (not params['DEBUG'] and not end):
        trackers.stats_list.append(copy(stats))

    # Save stats to file.
    if not params['DEBUG']:

        if stats['gen'] == 0:
            save_stats_headers(stats)

        save_stats_to_file(stats, end)

        if params['SAVE_ALL']:
            save_best_ind_to_file(stats, trackers.best_ever, end, stats['gen'])

        elif params['VERBOSE'] or end:
            save_best_ind_to_file(stats, trackers.best_ever, end)

    if end and not params['SILENT']:
        print_final_stats()


def get_moo_stats(individuals, end):
    """
    Generate the statistics for an evolutionary run with multiple objectives.
    Save statistics to utilities.trackers.stats_list. Print statistics. Save
    fitness plot information.

    :param individuals: A population of individuals for which to generate
    statistics.
    :param end: Boolean flag for indicating the end of an evolutionary run.
    :return: Nothing.
    """

    # Compute the pareto front metrics for the population.
    pareto = compute_pareto_metrics(individuals)

    # Save first front in trackers. Sort arbitrarily along first objective.
    trackers.best_ever = sorted(pareto.fronts[0], key=lambda x: x.fitness[0])

    # Store stats about pareto fronts.
    stats['pareto_fronts'] = len(pareto.fronts)
    stats['first_front'] = len(pareto.fronts[0])

    if end or params['VERBOSE'] or not params['DEBUG']:
        # Update all stats.
        update_stats(individuals, end)

    # Save fitness plot information
    if params['SAVE_PLOTS'] and not params['DEBUG']:

        # Initialise empty array for fitnesses for all inds on first pareto
        # front.
        all_arr = [[] for _ in range(params['FITNESS_FUNCTION'].num_obj)]

        # Generate array of fitness values.
        fitness_array = [ind.fitness for ind in trackers.best_ever]

        # Add paired fitnesses to array for graphing.
        for fit in fitness_array:
            for o in range(params['FITNESS_FUNCTION'].num_obj):
                all_arr[o].append(fit[o])

        if not end:
            trackers.first_pareto_list.append(all_arr)

            # Append empty array to best fitness list.
            trackers.best_fitness_list.append([])

            # Get best fitness for each objective.
            for o, ff in \
                    enumerate(params['FITNESS_FUNCTION'].fitness_functions):
                # Get sorted list of all fitness values for objective "o"
                fits = sorted(all_arr[o], reverse=ff.maximise)

                # Append best fitness to trackers list.
                trackers.best_fitness_list[-1].append(fits[0])

        if params['VERBOSE'] or end:

            # Plot best fitness for each objective.
            for o, ff in \
                    enumerate(params['FITNESS_FUNCTION'].fitness_functions):
                to_plot = [i[o] for i in trackers.best_fitness_list]

                # Plot fitness data for objective o.
                try:
                    #This is for objective function functions
                    plotname = ff.__name__ + str(o)
                except:
                    #This is for objective function classes
                    plotname = ff.__class__.__name__ + str(o)

                save_plot_from_data(to_plot, plotname)

            # TODO: PonyGE2 can currently only plot moo problems with 2
            #  objectives.
            # Check that the number of fitness objectives is not greater than 2
            if params['FITNESS_FUNCTION'].num_obj > 2:
                s = "stats.stats.get_moo_stats\n" \
                    "Warning: Plotting of more than 2 simultaneous " \
                    "objectives is not yet enabled in PonyGE2."
                print(s)

            else:
                save_pareto_fitness_plot()

    # Print statistics
    if params['VERBOSE'] and not end:
        print_generation_stats()
        print_first_front_stats()

    elif not params['SILENT']:
        # Print simple display output.
        perc = stats['gen'] / (params['GENERATIONS'] + 1) * 100
        stdout.write("Evolution: %d%% complete\r" % perc)
        stdout.flush()

    # Generate test fitness on regression problems
    if hasattr(params['FITNESS_FUNCTION'], "training_test") and end:

        for ind in trackers.best_ever:
            # Iterate over all individuals in the first front.

            # Save training fitness.
            ind.training_fitness = copy(ind.fitness)

            # Evaluate test fitness.
            ind.test_fitness = params['FITNESS_FUNCTION'](ind, dist='test')

            # Set main fitness as training fitness.
            ind.fitness = ind.training_fitness

    # Save stats to list.
    if params['VERBOSE'] or (not params['DEBUG'] and not end):
        trackers.stats_list.append(copy(stats))

    # Save stats to file.
    if not params['DEBUG']:

        if stats['gen'] == 0:
            save_stats_headers(stats)

        save_stats_to_file(stats, end)

        if params['SAVE_ALL']:
            save_first_front_to_file(stats, end, stats['gen'])

        elif params['VERBOSE'] or end:
            save_first_front_to_file(stats, end)

    if end and not params['SILENT']:
        print_final_moo_stats()

def get_soo_stats_v2(individuals, end):
    """
    Generate the statistics for an evolutionary run with a single objective.
    Save statistics to utilities.trackers.stats_list. Print statistics. Save
    fitness plot information.

    :param individuals: A population of individuals for which to generate
    statistics.
    :param end: Boolean flag for indicating the end of an evolutionary run.
    :return: Nothing.
    """

    # Get best individual.
    best = max(individuals)

    if not trackers.best_ever or best > trackers.best_ever:
        # Save best individual in trackers.best_ever.
        trackers.best_ever = best

    # Compute fitness on test for every generation
    if 'LISTENERS' in params:
        x_train = params['FITNESS_FUNCTION'].training_in
        x = x_train
        y_train = params['FITNESS_FUNCTION'].training_exp
        yhat_train = eval(trackers.best_ever.phenotype)

        for i in eval(params['LISTENERS']):
            trackers.aux['train_' + i.__name__ + "_best_ever"] = i(y_train, yhat_train)

    if hasattr(params['FITNESS_FUNCTION'], "training_test"):
        trackers.test_first_best_ever = params['FITNESS_FUNCTION'](trackers.best_ever, dist='test')

        x_test = params['FITNESS_FUNCTION'].test_in
        x_train = params['FITNESS_FUNCTION'].training_in
        y_test = params['FITNESS_FUNCTION'].test_exp
        y_train = params['FITNESS_FUNCTION'].training_exp
        if params['OPTIMIZE_CONSTANTS']:
            x = x_train
            yhat_train = eval(trackers.best_ever.phenotype_consec_consts)
            x = x_test
            yhat_test = eval(trackers.best_ever.phenotype_consec_consts)
        else:
            x = x_train
            try:
                yhat_train = eval(trackers.best_ever.phenotype)
            except:
                yhat_train = None
            x = x_test
            try:
                yhat_test = eval(trackers.best_ever.phenotype)
            except:
                yhat_test = None

        if 'LISTENERS' in params:
            for i in eval(params['LISTENERS']):
                trackers.aux['test_'+i.__name__+"_best_ever"] = i(y_test,yhat_test)

    if end or params['VERBOSE'] or not params['DEBUG']:
        # Update all stats.
        update_stats(individuals, end)

    # Save fitness plot information
    if params['SAVE_PLOTS'] and not params['DEBUG']:
        if not end:
            trackers.best_fitness_list.append(trackers.best_ever.fitness)

            if hasattr(params['FITNESS_FUNCTION'], "training_test"):
                if 'LISTENERS' in params:
                    for i in eval(params['LISTENERS']):
                        if not 'test_'+i.__name__+"_best_ever_list" in trackers.aux:
                            trackers.aux['test_'+i.__name__+"_best_ever_list"] = []
                            trackers.aux['train_' + i.__name__ + "_best_ever_list"] = []
                        trackers.aux['train_' + i.__name__ + "_best_ever_list"].append(trackers.aux['train_' + i.__name__ + "_best_ever"])

                        trackers.aux['test_' + i.__name__ + "_best_ever_list"].append(
                            trackers.aux['test_' + i.__name__ + "_best_ever"])
            else:
                if 'LISTENERS' in params:
                    for i in eval(params['LISTENERS']):
                        if not 'train_' + i.__name__ + "_best_ever_list" in trackers.aux:
                            trackers.aux['train_' + i.__name__ + "_best_ever_list"] = []
                        trackers.aux['train_' + i.__name__ + "_best_ever_list"].append(
                            trackers.aux['train_' + i.__name__ + "_best_ever"])

        if params['VERBOSE'] or end:
            save_plot_from_data(trackers.best_fitness_list, "best_fitness")

            if hasattr(params['FITNESS_FUNCTION'], 'training_test'):
                if 'LISTENERS' in params:
                    for i in eval(params['LISTENERS']):
                        to_plot = [[i,j] for i,j in zip(trackers.aux['train_'+i.__name__+"_best_ever_list"],
                                                        trackers.aux['test_'+i.__name__+"_best_ever_list"])]
                        save_plot_from_data(to_plot,
                                            i.__name__, label=['train', 'test'])
                        save_values_to_file(to_plot,
                                            i.__name__ + ".txt", header=['train', 'test'])
            else:
                if 'LISTENERS' in params:
                    for i in eval(params['LISTENERS']):
                        save_plot_from_data(trackers.aux['train_'+i.__name__+"_best_ever_list"],
                                            i.__name__, label=['train'])
                        save_values_to_file(trackers.aux['train_'+i.__name__+"_best_ever_list"],
                                            i.__name__ + ".txt", header=['train'])

    # Print statistics
    if params['VERBOSE'] and not end:
        print_generation_stats()

        if 'ASSOC_RULES_STATS' in params:
            print_assoc_rules_stats(individuals, params['ASSOC_RULES_STATS'])

    elif not params['SILENT']:
        # Print simple display output.
        perc = stats['gen'] / (params['GENERATIONS'] + 1) * 100
        stdout.write("Evolution: %d%% complete\r" % perc)
        stdout.flush()

    # Generate test fitness on regression problems
    if hasattr(params['FITNESS_FUNCTION'], "training_test") and end:
        # Save training fitness.
        trackers.best_ever.training_fitness = copy(trackers.best_ever.fitness)

        # Evaluate test fitness.
        trackers.best_ever.test_fitness = params['FITNESS_FUNCTION'](
            trackers.best_ever, dist='test')

        # Set main fitness as training fitness.
        trackers.best_ever.fitness = trackers.best_ever.training_fitness

    # Save stats to list.
    if params['VERBOSE'] or (not params['DEBUG'] and not end):
        trackers.stats_list.append(copy(stats))

    # Save stats to file.
    if not params['DEBUG']:

        if stats['gen'] == 0:
            save_stats_headers(stats)

        save_stats_to_file(stats, end)

        if params['SAVE_ALL']:
            save_best_ind_to_file(stats, trackers.best_ever, end, stats['gen'])

        elif params['VERBOSE'] or end:
            save_best_ind_to_file(stats, trackers.best_ever, end)

    if end and not params['SILENT']:
        print_final_stats()

def get_moo_stats_v2(individuals, end):
    """
    Generate the statistics for an evolutionary run with multiple objectives.
    Save statistics to utilities.trackers.stats_list. Print statistics. Save
    fitness plot information.

    This second version just uses compute_pareto_metrics_v2 instead of compute_pareto_metrics. FIXME In addition, it also computes fitness values on test dataset per generation, and outputs these values

    :param individuals: A population of individuals for which to generate
    statistics.
    :param end: Boolean flag for indicating the end of an evolutionary run.
    :return: Nothing.
    """

    # Compute the pareto front metrics for the population.
    pareto = compute_pareto_metrics_v2(individuals)

    # Save first front in trackers. Sort arbitrarily along first objective.
    trackers.best_ever = sorted(pareto.fronts[0], key=lambda x: x.fitness[0])

    # Compute fitness on test for every generation
    if hasattr(params['FITNESS_FUNCTION'], "training_test"):
        trackers.test_first_best_ever = params['FITNESS_FUNCTION'](trackers.best_ever[0], dist='test')

    # Store stats about pareto fronts.
    stats['pareto_fronts'] = len(pareto.fronts)
    stats['first_front'] = len(pareto.fronts[0])

    if end or params['VERBOSE'] or not params['DEBUG']:
        # Update all stats.
        update_stats(individuals, end)

    # Save fitness plot information
    if params['SAVE_PLOTS'] and not params['DEBUG']:

        # Initialise empty array for fitnesses for all inds on first pareto
        # front.
        all_arr = [[] for _ in range(params['FITNESS_FUNCTION'].num_obj)]

        # Generate array of fitness values.
        fitness_array = [ind.fitness for ind in trackers.best_ever]

        # Add paired fitnesses to array for graphing.
        for fit in fitness_array:
            for o in range(params['FITNESS_FUNCTION'].num_obj):
                all_arr[o].append(fit[o])

        if not end:
            trackers.first_pareto_list.append(all_arr)

            # Append empty array to best fitness list.
            trackers.best_fitness_list.append([])

            # Get best fitness for each objective.
            for o, ff in \
                    enumerate(params['FITNESS_FUNCTION'].fitness_functions):
                # Get sorted list of all fitness values for objective "o"
                fits = sorted(all_arr[o], reverse=ff.maximise)

                # Append best fitness to trackers list.
                if hasattr(params['FITNESS_FUNCTION'], "training_test"):
                        trackers.best_fitness_list[-1].append([fits[0], trackers.test_first_best_ever[o]])
                else:
                    trackers.best_fitness_list[-1].append(fits[0])

        if params['VERBOSE'] or end:

            # Plot best fitness for each objective.
            for o, ff in \
                    enumerate(params['FITNESS_FUNCTION'].fitness_functions):
                to_plot = [i[o] for i in trackers.best_fitness_list]

                # Plot fitness data for objective o.
                try:
                    #This is for objective function functions
                    plotname = ff.__name__ + str(o)
                except:
                    #This is for objective function classes
                    plotname = ff.__class__.__name__ + str(o)

                if hasattr(params['FITNESS_FUNCTION'], "training_test"):
                    save_plot_from_data(to_plot, plotname, label=['train', 'test'])
                    save_values_to_file(to_plot, plotname+".txt", ['train', 'test'])
                else:
                    save_plot_from_data(to_plot, plotname)
                    save_values_to_file(to_plot, plotname + ".txt")

            # TODO: PonyGE2 can currently only plot moo problems with 2
            #  objectives.
            # Check that the number of fitness objectives is not greater than 2
            if params['FITNESS_FUNCTION'].num_obj > 2:
                s = "stats.stats.get_moo_stats\n" \
                    "Warning: Plotting of more than 2 simultaneous " \
                    "objectives is not yet enabled in PonyGE2."
                print(s)

            else:
                save_pareto_fitness_plot()

    # Print statistics
    if params['VERBOSE'] and not end:
        print_generation_stats()
        print_first_front_stats()

    elif not params['SILENT']:
        # Print simple display output.
        perc = stats['gen'] / (params['GENERATIONS'] + 1) * 100
        stdout.write("Evolution: %d%% complete\r" % perc)
        stdout.flush()

    # Generate test fitness on regression problems
    if hasattr(params['FITNESS_FUNCTION'], "training_test") and end:

        for ind in trackers.best_ever:
            # Iterate over all individuals in the first front.

            # Save training fitness.
            ind.training_fitness = copy(ind.fitness)

            # Evaluate test fitness.
            ind.test_fitness = params['FITNESS_FUNCTION'](ind, dist='test')

            # Set main fitness as training fitness.
            ind.fitness = ind.training_fitness

    # Save stats to list.
    if params['VERBOSE'] or (not params['DEBUG'] and not end):
        trackers.stats_list.append(copy(stats))

    # Save stats to file.
    if not params['DEBUG']:

        if stats['gen'] == 0:
            save_stats_headers(stats)

        save_stats_to_file(stats, end)

        if params['SAVE_ALL']:
            save_first_front_to_file(stats, end, stats['gen'])

        elif params['VERBOSE'] or end:
            save_first_front_to_file(stats, end)

    if end and not params['SILENT']:
        print_final_moo_stats()

def update_stats(individuals, end):
    """
    Update all stats in the stats dictionary.

    :param individuals: A population of individuals.
    :param end: Boolean flag for indicating the end of an evolutionary run.
    :return: Nothing.
    """

    if not end:
        # Time Stats
        trackers.time_list.append(time() - stats['time_adjust'])
        stats['time_taken'] = trackers.time_list[-1] - \
            trackers.time_list[-2]
        stats['total_time'] = trackers.time_list[-1] - \
            trackers.time_list[0]

    # Population Stats
    stats['total_inds'] = params['POPULATION_SIZE'] * (stats['gen'] + 1)
    stats['runtime_error'] = len(trackers.runtime_error_cache)
    if params['CACHE']:
        stats['unique_inds'] = len(trackers.cache)
        stats['unused_search'] = 100 - stats['unique_inds'] / \
            stats['total_inds'] * 100

    # Genome Stats
    genome_lengths = [len(i.genome) for i in individuals]
    stats['max_genome_length'] = np.nanmax(genome_lengths)
    stats['ave_genome_length'] = np.nanmean(genome_lengths)
    stats['min_genome_length'] = np.nanmin(genome_lengths)

    # Used Codon Stats
    codons = [i.used_codons for i in individuals]
    stats['max_used_codons'] = np.nanmax(codons)
    stats['ave_used_codons'] = np.nanmean(codons)
    stats['min_used_codons'] = np.nanmin(codons)

    # Tree Depth Stats
    depths = [i.depth for i in individuals]
    stats['max_tree_depth'] = np.nanmax(depths)
    stats['ave_tree_depth'] = np.nanmean(depths)
    stats['min_tree_depth'] = np.nanmin(depths)

    # Tree Node Stats
    nodes = [i.nodes for i in individuals]
    stats['max_tree_nodes'] = np.nanmax(nodes)
    stats['ave_tree_nodes'] = np.nanmean(nodes)
    stats['min_tree_nodes'] = np.nanmin(nodes)

    if not hasattr(params['FITNESS_FUNCTION'], 'multi_objective'):
        # Fitness Stats
        fitnesses = [i.fitness for i in individuals if i.fitness is not None]
        stats['ave_fitness'] = np.nanmean(fitnesses, axis=0)
        stats['best_fitness'] = trackers.best_ever.fitness


def print_generation_stats():
    """
    Print the statistics for the generation and individuals.

    :return: Nothing.
    """

    print("______\n")
    for stat in sorted(stats.keys()):
        print(" ", stat, ": \t", stats[stat])
    print("\n")


def print_first_front_stats():
    """
    Stats printing for the first pareto front for multi-objective optimisation.

    :return: Nothing.
    """

    print("  first front fitnesses :")
    for ind in trackers.best_ever:
        print("\t  ", ind.fitness)


def print_final_stats():
    """
    Prints a final review of the overall evolutionary process.

    :return: Nothing.
    """

    if hasattr(params['FITNESS_FUNCTION'], "training_test"):
        print("\n\nBest:\n  Training fitness:\t",
              trackers.best_ever.training_fitness)
        print("  Test fitness:\t\t", trackers.best_ever.test_fitness)
    else:
        print("\n\nBest:\n  Fitness:\t", trackers.best_ever.fitness)

    print("  Phenotype:", trackers.best_ever.phenotype)
    print("  Genome:", trackers.best_ever.genome)
    print_generation_stats()


def print_final_moo_stats():
    """
    Prints a final review of the overall evolutionary process for
    multi-objective problems.

    :return: Nothing.
    """

    print("\n\nFirst Front:")
    for ind in trackers.best_ever:
        print(" ", ind)
    print_generation_stats()

def print_assoc_rules_stats(individuals, filename):
    # Obtein and evaluate the rules
    rules = RuleSet()

    for ind in individuals:
        if ind.invalid == False:
            rules.read_tree(ind.tree)

    if 'ASSOC_CONF_FILTER' not in params:
        conf_filtering = 0.7
    else:
        conf_fitlering = params['ASSOC_CONF_FILTER']

    # Filter the rules
    if 'FILTEROUT_NOTTARGET_RULES' in params and params['FILTEROUT_NOTTARGET_RULES']:
        rules.filter_by_consequent("'"+params['CLASS_ASSOC_RULES_TARGET']+"'")

    rules.filter_duplicates()
    rules.filter_by_confidence(params['FITNESS_FUNCTION'].training_in,
                               params['FITNESS_FUNCTION'].training_exp, conf_fitlering)

    # Evaluación
    with open(path.join(params['FILE_PATH'],filename), "a") as f:
        AssocRules_Stats().print_stats(rules.rules,
                                   params['FITNESS_FUNCTION'].training_in,
                                   params['FITNESS_FUNCTION'].training_exp, f)
