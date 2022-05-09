import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric

from algorithm.parameters import params

def get_feature_importances_from_inds(individuals):
    importances = []
    features = params['FITNESS_FUNCTION'].training_in.columns
    num_features = len(features)
    null_importances = np.zeros(num_features)

    for ind in individuals:
        # Just for valid individuals; invalid individuals has phenotype = None and fitness = nan.
        if ind.invalid != True and np.isnan(ind.fitness) != True:
            importances.append(get_features_importances(ind.tree))
        else:
            importances.append(null_importances)

    return importances

def get_features_importances(tree, importances = None):
    features = params['FITNESS_FUNCTION'].training_in.columns

    if not isinstance(importances, np.ndarray) and importances == None:
        num_features = len(features)
        importances = np.zeros(num_features)
        importances, _ = get_features_importances(tree, importances)
        return importances
    else:
        if tree.children[0].root == 'np.where(':

            _, this_condition, _, _, \
            _ = tree.children[1].get_tree_info(params['BNF_GRAMMAR'].non_terminals.keys(), [], [])
            these_features = [(index,i) for index,i in enumerate(features) for j in this_condition if '\''+i+'\'' in j]

            importances, num_leaves_left = get_features_importances(tree.children[3], importances)
            importances, num_leaves_right = get_features_importances(tree.children[5], importances)
            num_leaves = num_leaves_right + num_leaves_left

            for feature in [i[0] for i in these_features]:
                importances[feature] += num_leaves
            return importances, num_leaves
        else:
            return importances, 1



def get_features_indexes(phenotype):
    """
    Function that gets the features indexes used by an individual, by handling its
    phenotype (string).

    Parameters
    ----------
    - phenotype: string to be handled.

    Returns
    -------
    - result: sorted list with the wanted-unique indexes.
    """

    ############ CÓDIGO CARLOS##############
    attrs = phenotype.split('x[\'')[1:]
    attrs = [j for i in attrs for j in i.split('\']')][::2]

    # phenotype_splitted = phenotype.split('\'')
    used_attrs = [i in attrs for i in params['FITNESS_FUNCTION'].training_in.columns] # TODO esto hay que multiplicarlo por 2^n con n la profundidad del subarbol donde aparece dicho atributo
    return np.where(used_attrs)[0]
    ############ FIN CÓDIGO CARLOS##########

    ############# CÓDIGO VENTURA##############
    # subphenotype = phenotype.split("x.iloc")
    #
    # # Check if 'subphenotype' string has brackets.
    # for i in range(len(subphenotype) - 1):
    #     if subphenotype[i].find("[") == -1:
    #         subphenotype.pop(i)
    #
    # # Get the unformatted features indexes.
    # features = []
    # for i in subphenotype:
    #     opening_bracket = i.index("[")
    #     closing_bracket = i.index("]")
    #     features.append(i[opening_bracket:closing_bracket + 1])
    #
    # result = []
    # for feature in features:
    #     # Get the integers, which reference each feature.
    #     result.append(re.findall(r'\d+', feature))
    #
    # # Flatten the result.
    # result = [int(item) for sublist in result for item in sublist]
    #
    # # Get sorted wanted-unique indexes.
    # #   To make them unique --> list(set(result)).
    # return sorted(list(set(result)))
    ##########FIN CÓDIGO VENTURA##################3


def get_ind_used_features(individuals, n_dataset_features):
    """
    Function that provides the features used by each individual (headers of a dataset).

    Parameters
    ----------
    - individuals: population on which to obtain the characteristics.
    - n_dataset_features: amount of columns in the dataset.

    Returns
    -------
    - features: list with the index of the individual and the used features. These used
    features is a binary list within the 'features' list. The architecture of the list
    is as follows: [individual, [0, 1, 1, 0, 1, 0, ..., 0]].
    """
    features = []
    for index, ind in enumerate(individuals):
        # Just for valid individuals; invalid individuals has phenotype = None and fitness = nan.
        if ind.invalid != True and np.isnan(ind.fitness) != True:
            # Binary features array.
            binary = np.zeros(n_dataset_features)
            # Feature indexes list.
            feature_indexes = get_features_indexes(ind.phenotype)

        # FIXME: Quizás tabular líneas 74, 75 y 77, porque si empieza por uno inválido,
        # puede lanzar un Exception.
        for i in feature_indexes:
            binary[i] = 1

        features.append([index, binary])

    return features


def compute_ind_similarities(features, n_trees):
    """
    Funtion that computes the similarity between individuals by using the used
    features. `cosine_similarity` by default.

    Parameters
    ----------
    - features: list with the index of the individual and the used features. These used
    features is a binary list within the 'features' list. The architecture of the list is
    divided in two parts as follows: [individual, [0, 1, 1, 0, 1, 0, ..., 0]].
    - n_trees: amount of individuals.

    Returns
    -------
    - A: similarity matrix.
    - all_similarities: similarity array (flattened and sorted similarity matrix).
    """
    similarities = []
    all_similarities = []

    # Fill the matrix with the features (The second part of the list).
    for i in range(n_trees):
        similarities.append(features[i][1].tolist())

    # Compute similarity matrix over the features.
    A = cosine_similarity(similarities)

    # Print similarity matrix.
    # print('pairwise dense output:\n {}\n'.format(A))

    # Get all similarities in one sorted list.
    for i in range(n_trees):
        for j in range(i):
            all_similarities.append(A[i][j])
    all_similarities.sort()

    return A, all_similarities


def new_similarity_approach(features, n_trees):
    """
    Funtion that computes the similarity between individuals by using the used
    features.

    Parameters
    ----------
    - features: list with the index of the individual and the used features. These used
    features is a binary list within the 'features' list. The architecture of the list is
    divided in two parts as follows: [individual, [0, 1, 1, 0, 1, 0, ..., 0]].
    - n_trees: amount of individuals.

    Returns
    -------
    - A: similarity matrix.
    - all_similarities: similarity array (flattened and sorted similarity matrix).
    """
    similarities = []
    all_similarities = []

    # Fill the matrix with the features (The second part of the list).
    for i in range(n_trees):
        similarities.append(features[i][1].tolist())

    # Count the ones from each list.
    ones = [i.count(1) for i in similarities]

    # Compute similarity matrix over the features.
    #A = []  # Similarity matrix. #FIXED
    A = np.zeros((n_trees,n_trees))
    for i in reversed(range(len(similarities))):
        # aux = []
        for j in reversed(range(i)):# FIXED, len(similarities))):
            if i == j:
                # Same element, similarity = 1.
                A[i,i] = 1
                # aux.append(1)
            else:
                # Boolean array of common elements:
                # 0: is not common / 1: is common.
                commons = [1 if tuple(item)[0] == 1 and tuple(
                    item)[1] == 1 else 0 for item in zip(similarities[i], similarities[j])] #FIXED Antes contaba todas las coincidencias

                # Count the amount of commons.
                n_commons = commons.count(1)

                # Select the minimun amount of ones between list 'i' and 'j'
                if i == j:
                    minimum = ones[i]
                else:
                    minimum = min(ones[i], ones[j])

                # Append to the similarity matrix the similarity factor.
                if minimum > 0:
                    A[i,j] = n_commons/minimum
                    A[j,i] = A[i,j]
                    # aux.append(n_commons/minimum)
                else:
                    A[i,j] = 0
                    A[j,i] = A[i,j]
                    # aux.append(0)
        # A.append(aux)

    # Print similarity matrix.
    # print('pairwise dense output:\n {}\n'.format(A))

    # Get all similarities in one sorted list.
    # for i in range(n_trees): #FIXED
    #     for j in range(i):
    #         all_similarities.append(A[i][j])
    # all_similarities.sort()

    # all_similarities = [i for j in A for i in j]

    all_similarities = A.flatten()

    return A, all_similarities


def compute_ind_distances(features, n_trees):
    """
    Funtion that computes the distance between individuals by using the used
    features. `Hamming distance` by default.

    Parameters
    ----------
    - features: list with the index of the individual and the used features. These used
    features is a binary list within the 'features' list. The architecture of the list is
    divided in two parts as follows: [individual, [0, 1, 1, 0, 1, 0, ..., 0]].
    - n_trees: amount of individuals.

    Returns
    -------
    - A: similarity matrix.
    - all_distances: distance array (flattened and sorted distance matrix).
    """
    distances = []
    all_distances = []

    # Fill the matrix with the features (The second part of the list).
    for i in range(n_trees):
        distances.append(features[i][1].tolist())

    # Compute distance matrix over the features.
    dist = DistanceMetric.get_metric("hamming")
    A = dist.pairwise(distances)

    # Print distance matrix.
    # print('pairwise dense output:\n {}\n'.format(A))

    # Get all distances in one sorted list.
    for i in range(n_trees):
        for j in range(i):
            all_distances.append(A[i][j])
    all_distances.sort()

    return A, all_distances


def compute_crowding(A, A_flat, n_trees, procedure):
    """
    Function that computes a crowding vector by using the .25/.75 percentile as
    threshold of a sorted distance/similarity list, depending on the selected procedure.

    Parameters
    ----------
    - A: similarity/distance matrix.
    - A_flat: similarity/distance array (flattened matrix).
    - n_trees: amount of individuals.
    - procedure: select between `"similarity"` or `"distance"`.

    Returns
    -------
    - crowding: array with the penality associated to each individual's fitness.
    """
    if procedure == "distance":
        # Use percentile .25 as threshold.
        p_25 = np.percentile(A_flat, 25)
        p_25 = np.max(A_flat)

        crowding = np.zeros(n_trees)
        for i in reversed(range(n_trees)):
            for j in reversed(range(i)):
                if A[i][j] <= p_25:
                    crowding[i] += 1

    if procedure == "similarity":
        # Use percentile .75 as threshold.
        p_75 = np.percentile(A_flat, 75)

        crowding = np.zeros(n_trees)
        for i in reversed(range(n_trees)):
            for j in reversed(range(i)):
                if A[i][j] >= p_75:
                    crowding[i] += 1

    return crowding


def update_fitness(individuals, crowding, n_trees):
    """
    Function that updates the fitness asociated to each individual by using
    crowding.

    Parameters
    ----------
    - individuals: population which individuals' fitness is going to be updated.
    - crowding: array with the rate-reduction of each individuals' fitness.
    - n_trees: amount of individuals.

    Returns
    -------
    Nothing.
    """
    # Since the last fitness are 'np.nan', we have to search where is the first
    # last numerical value. This value is set as the penalty.
    for ind in individuals[::-1]:
        if np.isnan(ind.fitness) == False:
            penalty = ind.fitness
            break

    # Update fitness.
    values = []
    for i in range(n_trees):
        # Some individuals are not valid, so that, they have fitness = 'np.nan'.
        # Just taking in account those whose fitness is a number.
        if not np.isnan(individuals[i].fitness):
            individuals[i].fitness += (crowding[i] * penalty)
            values.append(individuals[i].fitness)
            # if crowding[i] > 0:
            #     individuals[i].fitness = np.nan
            #     individuals[i].invalid = True

    # print('VALUES', np.mean(np.array(values)))


def diversification(individuals):
    """
    Function that provides diversification to the population of individuals
    by computing the cosine similarity of the used features of each individual,
    then computing the crowding and finally, updating the fitness asociated to
    each individual using the previous crowding.

    Parameters
    ----------
    - individuals: population to be diversificated.

    Returns
    -------
    - individuals: population diversificated.
    """
    # Descending sort of individuals.
    individuals.sort(reverse=True)

    # Get training and test data.
    training_in, training_exp, test_in, test_exp = params[
        'FITNESS_FUNCTION'].training_in, params['FITNESS_FUNCTION'].training_exp, params['FITNESS_FUNCTION'].test_in, params['FITNESS_FUNCTION'].test_exp

    x = training_in
    y = training_exp

    # Get the features of each individual.
    n_dataset_features = len(x.iloc[0, :])
    n_trees = len(individuals)

    if params['SHARING_PROCEDURE'] == 'importance disimilarity':
        importances = get_feature_importances_from_inds(individuals)
        A, A_flat = ratio_new_rules(importances)
        crowding = compute_crowding(A, A_flat, n_trees, procedure='distance')
    else:
        features = get_ind_used_features(individuals, n_dataset_features)

        # Compute distances/similarities. Use params["SHARING_PROCEDURE"] for setting
        # the parameter: "distance" for distance matrix and "similarity" for similarity
        # matrix.
        if params["SHARING_PROCEDURE"] == "distance":
            A, A_flat = compute_ind_distances(features, n_trees)
        if params["SHARING_PROCEDURE"] == "similarity":
            #A, A_flat = compute_ind_similarities(features, n_trees)
            A, A_flat = new_similarity_approach(features, n_trees)

        # Compute crowding.
        crowding = compute_crowding(
            A, A_flat, n_trees, procedure=params["SHARING_PROCEDURE"])

    # Update fitness.
    update_fitness(individuals, crowding, n_trees)

    return individuals

def ratio_new_rules(importances):
    num_trees = len(importances)
    A = np.zeros((num_trees, num_trees))

    for i, imp_i in zip(range(num_trees), importances):
        for j, imp_j in zip(range(i+1,num_trees), importances[i+1:]):
            differences = np.sum(np.abs(imp_i - imp_j))
            #max_num_rules = min(np.sum(imp_i), np.sum(imp_j))
            max_num_rules = 1
            try:
                A[i,j] = differences / max_num_rules
            except:
                A[i,j] = 0
            A[j,i] = A[i,j]

    return A, A.flatten()