from copy import copy
from sys import stdout
from time import time
from os import path

import numpy as np
import pandas as pd
from algorithm.parameters import params
from utilities.algorithm.NSGA2 import compute_pareto_metrics
from utilities.algorithm.state import create_state
from utilities.stats import trackers
from utilities.stats.file_io import save_best_ind_to_file, \
    save_first_front_to_file, save_stats_headers, save_stats_to_file
from utilities.stats.save_plots import save_pareto_fitness_plot, \
    save_plot_from_data

from utilities.fitness.assoc_rules_measures import get_cond_in_antec, get_attr_stats, get_metrics, get_min_avg_confidence, get_min_avg_support, get_uncovered_patterns, update_metrics
from utilities.misc.nested_conds_2_rules_list import nested_conds_2_rules_list
import matplotlib.pyplot as plt


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
        get_moo_stats(individuals, end)

    else:
        # Single objective optimisation is being used.
        get_soo_stats(individuals, end)

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

    # Get the best 'params['ELITE_SIZE']' individuals of the population.
    if end and params['ELITE_SIZE'] != None:
        if params['ELITE_SIZE'] > 1:
            get_pop_metrics(individuals, params['ELITE_SIZE'])


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
        fitnesses = [i.fitness for i in individuals]
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


def get_pop_metrics(individuals, elite_size):
    """
    Prints the final elite or population of the evolutionary process and its metrics.

    Parameters
    ----------
    individuals: list
        Individuals of the final population.
    elite_size: integer
        Size of the elite (Preselected in parameters as 'ELITE_SIZE').

    Returns
    -------
    Nothing.
    """

    individuals.sort(reverse=True)

    phenotype_list = []
    fitness_list = []

    phenotype_index_list = []
    rule_list = []

    antec_support_list = []
    consec_support_list = []
    rule_support_list = []

    precision_list = []
    recall_list = []
    lift_list = []
    leverage_list = []
    conviction_list = []

    covered_targets_list = []

    for i in range(len(individuals)):  # elite_size):
        # Note: this line is required if you want to save all the population to avoid invalid and
        # fitness = nan individuals.
        if individuals[i].invalid != True and np.isnan(individuals[i].fitness) != True:
            # Compare each individual with the others.
            for j in range(i + 1, len(individuals)):
                if individuals[i].phenotype == individuals[j].phenotype:
                    # If the individual is repeated, set it as 'invalid'.
                    individuals[j].invalid = True

            # Save non-repeated individuals.
            fitness_list.append(individuals[i].fitness)
            phenotype_list.append(individuals[i].phenotype)

            elite_metrics = get_phenotype_metrics(individuals[i].phenotype, i)

            phenotype_index_list.append(elite_metrics[0])
            rule_list.append(elite_metrics[1])
            antec_support_list.append(elite_metrics[2])
            consec_support_list.append(elite_metrics[3])
            rule_support_list.append(elite_metrics[4])
            precision_list.append(elite_metrics[5])
            recall_list.append(elite_metrics[6])
            lift_list.append(elite_metrics[7])
            leverage_list.append(elite_metrics[8])
            conviction_list.append(elite_metrics[9])
            covered_targets_list.append(elite_metrics[10])

    # Flatten the lists. Note that in some cases (with complex rules) it is necessary.
    flat_phenotype_index_list = [
        item for sublist in phenotype_index_list for item in sublist]
    flat_rule_list = [item for sublist in rule_list for item in sublist]

    # Note: Extra metrics.
    flat_antec_support_list = [
        item for sublist in antec_support_list for item in sublist]
    flat_consec_support_list = [
        item for sublist in consec_support_list for item in sublist]
    flat_rule_support_list = [
        item for sublist in rule_support_list for item in sublist]

    flat_precision_list = [
        item for sublist in precision_list for item in sublist]
    flat_recall_list = [item for sublist in recall_list for item in sublist]
    flat_lift_list = [item for sublist in lift_list for item in sublist]
    flat_leverage_list = [
        item for sublist in leverage_list for item in sublist]
    flat_conviction_list = [
        item for sublist in conviction_list for item in sublist]

    flat_covered_targets_list = [
        item for sublist in covered_targets_list for item in sublist]

    # Dataframes with results.
    #####################################################################
    # Trees dataframe:
    #####################################################################
    df_trees = pd.DataFrame(list(zip(phenotype_list, fitness_list)),
                            columns=["Phenotype", "Fitness"])

    # Print and save data into '.csv' file at 'results' path.
    print(df_trees)
    filename = path.join(params['FILE_PATH'], "trees.csv")
    df_trees.to_csv(filename)
    # print(df_trees.to_latex(index=True))

    #####################################################################
    # Metrics dataframes:
    #####################################################################

    df_metrics = pd.DataFrame(list(zip(flat_phenotype_index_list, flat_rule_list, flat_precision_list, flat_recall_list, flat_lift_list, flat_leverage_list, flat_conviction_list, flat_antec_support_list, flat_consec_support_list, flat_rule_support_list, flat_covered_targets_list)),
                              columns=["Phenotype", "Rule", "Precision", "Recall", "Lift", "Leverage", "Conviction", "S(A)", "S(C)", "S(A-->C)", "Covered patterns"])

    # We want non-repeated rules.
    df_metrics = df_metrics.drop_duplicates(subset=["Rule"])

    print()
    print(df_metrics)

    # Save metrics.
    filename = path.join(params['FILE_PATH'], "metrics.csv")
    df_metrics.to_csv(filename, index=False)
    # print(df_metrics.to_latex(index=False))

    # Update metrics given a condition. Read brief for more information.
    df_metrics_updated = update_metrics(df_metrics)

    # Save updated metrics.
    filename = path.join(params['FILE_PATH'], "metrics_updated.csv")
    df_metrics_updated.to_csv(filename, index=False)

    # NOTE: Filtering out some of these metrics:
    # Discard association rules with consecuent = No.
    df_metrics_updated_filtered = df_metrics_updated[df_metrics_updated["Rule"].str.contains(
        "--> No") == False]

    # Discard association rules with precision < 0.7.
    df_metrics_updated_filtered = df_metrics_updated_filtered.drop(
        df_metrics_updated_filtered[df_metrics_updated_filtered["Precision"] < 0.7].index)

    # Save updated metrics.
    filename = path.join(params['FILE_PATH'], "metrics_updated_filtered.csv")
    df_metrics_updated_filtered.to_csv(filename, index=False)

    #####################################################################
    # Extra metrics:
    #####################################################################

    # % Support (Minimum and average support):
    lhs = get_min_avg_support(df_metrics_updated_filtered["S(A)"])

    # % Confidence (minimum and average confidence):
    conf = get_min_avg_confidence(df_metrics_updated_filtered["Precision"])

    # N rules (Number of rules):
    n_rules = len(df_metrics_updated_filtered)

    # N antecedent conditions (compute the number of conditions within an antecedent):
    n_antecedents, antec_counts = get_cond_in_antec(
        df_metrics_updated_filtered["Rule"])

    # Get the amount of rows and features in dataset. This will be used to compute
    # the 'use_freq'.
    n_rows, n_features = params['FITNESS_FUNCTION'].training_in.shape[0], params['FITNESS_FUNCTION'].training_in.shape[1]

    # Used attributes in rules:
    used_attr, use_freq = get_attr_stats(
        df_metrics_updated_filtered["Rule"], n_features)

    # Uncovered patterns:
    uncovered_patterns = get_uncovered_patterns(
        df_metrics_updated_filtered["Covered patterns"])

    df_extra = pd.DataFrame([[lhs, conf, n_rules, antec_counts, used_attr, use_freq, uncovered_patterns]],
                            columns=[f"%LHS.Sup.\nMin:Avg", "%Conf.\nMin:Avg", "#rules", " |antecedent|\n" + str(n_antecedents), "Used attrs.", "%Attr.use freq.", "%Unc.pos."])

    # Save extra metrics.
    filename = path.join(params['FILE_PATH'], "metrics_extra.csv")
    df_extra.to_csv(filename, index=False)

    # Plot average fitness of the whole evolutionary process.
    plot_elite_average_fitness()


def get_phenotype_metrics(phenotype, phenotype_index):
    """
    Read a phenotype and compute the following metrics: precision, recall, lift, leverage and conviction.

    Parameters
    ----------
    phenotype: Phenotype of an individual.
    phenotype_index: Index of the current phenotype.

    Returns
    -------
    List with all the computed metrics as follows: [antec_support, consec_support, rule_support, rule_precision, rule_recall, rule_lift, rule_leverage, rule_conviction].
    """

    # Get training and test data
    training_in, training_exp, test_in, test_exp = params[
        'FITNESS_FUNCTION'].training_in, params['FITNESS_FUNCTION'].training_exp, params['FITNESS_FUNCTION'].test_in, params['FITNESS_FUNCTION'].test_exp

    x = training_in
    y = training_exp

    rules, consecuents = nested_conds_2_rules_list(phenotype)

    rules_length = len(rules)

    assert rules_length == len(
        consecuents), "Length of 'rules' list and 'consecuents' list must be the same."

    # Elements to return:
    phenotype_index_list = []
    rule_list = []

    antec_support_list = []
    consec_support_list = []
    rule_support_list = []

    precision_list = []
    recall_list = []
    lift_list = []
    leverage_list = []
    conviction_list = []

    covered_targets_list = []

    for index in range(rules_length):
        # Get the list of metrics.
        metrics = get_metrics(
            eval(rules[index]), y, consecuents[index], visualize=False)

        phenotype_index_list.append(phenotype_index)
        rule_list.append(rules[index] + " --> " + consecuents[index])
        antec_support_list.append(metrics[0])
        consec_support_list.append(metrics[1])
        rule_support_list.append(metrics[2])
        precision_list.append(metrics[3])
        recall_list.append(metrics[4])
        lift_list.append(metrics[5])
        leverage_list.append(metrics[6])
        conviction_list.append(metrics[7])

        covered_targets_list.append(metrics[8])

    return [phenotype_index_list, rule_list, antec_support_list, consec_support_list, rule_support_list, precision_list, recall_list, lift_list, leverage_list, conviction_list, covered_targets_list]


def get_elite_average_fitness(current_elite):
    """
    Function that get the average fitness of each elite in the whole
    evolutionary process. The average fitness is saved in a '.txt'
    file inside 'results' folder.

    Parameters
    ----------
    - current_elite: elite in the i-th iteration.

    Returns
    -------
    Nothing.
    """
    sum = 0
    for ind in current_elite:
        sum += ind.fitness

    average_elite_fitness = str(sum / len(current_elite))

    file_name = path.join(params['FILE_PATH'], "average_fitness.txt")
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(average_elite_fitness)


def plot_elite_average_fitness():
    """
    Plot average fitness. The result is saved as '.png' at 'results' folder.
    """

    # Average fitness:
    with open(path.join(params['FILE_PATH'], "average_fitness.txt")) as f:
        lines = f.readlines()

    # Average fitness list.
    y = []
    for line in lines:
        y.append(float(line))

    # Generations:
    #x = [i for i in range(len(y))]
    x = range(len(y))

    plt.ylabel('average_fitness')
    plt.xlabel('Generation')
    plt.plot(x, y)

    # # Change fontsize according to the number of elements of x-axis.
    # fontsize = 500 / len(x)
    # if fontsize > 10:
    #     fontsize = 10

    # # Force x-axis (Generations) to be integer.
    # plt.xticks(x, fontsize=fontsize)

    # Save plot as '.png' at 'results' folder.
    plt.savefig(path.join(params['FILE_PATH'], "average_fitness.png"))
