import statistics
import numpy as np
import pandas as pd
from algorithm.parameters import params
from fitness.supervised_learning.diversification import get_features_indexes


def get_metrics(antec_eval, y, consec, visualize):
    """
    Function that computes:
        - Antecedent support.
            S(A) = P(A)
        - Consecuent support.
            S(C) = P(C)
        - Rule support:
            S(A --> C) = S(A union C), range: [0, 1]
        - Rule precision.
            Precision = tp / (tp + fp)
        - Rule recall.
            Recall = tp / (tp + fn)
        - Rule lift.
            Lift(A --> C) = Confidence(A --> C) / Support(C), range: [0, inf]
        - Rule leverage.
            Leverage(A --> C) = Support(A --> C) - Support(A) x Support(C), range: [-1, 1]
        - Rule conviction.
            Conviction(A --> C) = (1 - Support(C)) / (1 - Confidence(A --> C)), range: [0, inf]

    :param antec_eval: Boolean array with the evaluation of the rule in the dataset.
    :param y: set of targets.
    :param consec: Current consecuent.
    :param visualize: Boolean to show the metrics.
    :return list with all the computed metrics as follows: [antec_support, consec_support, rule_support, rule_precision, rule_recall, rule_lift, rule_leverage, rule_conviction, covered_patterns_list].
    """
    covered_targets = []
    try: # If we try to get rows with (-1.0 <= (-3.0 + -0.01)), which can be generated, that would produce an error
        covered_targets = y[antec_eval]
    except:
        pass

    # If there are no covered targets, return 'np.nan'.
    if len(covered_targets) < 1:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    antec_support = sum(antec_eval) / len(antec_eval)
    consec_support = sum(y == consec) / len(y)
    rule_support = sum(covered_targets == consec) / len(y)

    rule_precision = sum(covered_targets == consec) / len(covered_targets)
    rule_recall = sum(covered_targets == consec) / sum(y == consec)
    rule_lift = rule_precision / consec_support
    rule_leverage = rule_support - antec_support * consec_support

    if rule_precision == 1.0:
        rule_conviction = np.inf
    else:
        rule_conviction = (1.0 - consec_support) / (1.0 - rule_precision)

    if visualize == True:
        print(f"\t{antec_support=}")
        print(f"\t{consec_support=}")
        print(f"\t{rule_support=}")
        print()
        print(f"\t{rule_precision=}")
        print(f"\t{rule_recall=}")
        print(f"\t{rule_lift=}")
        print(f"\t{rule_leverage=}")
        print(f"\t{rule_conviction=}")
        print()

    covered_patterns_list = covered_targets.index.tolist()

    return [antec_support, consec_support, rule_support, rule_precision, rule_recall, rule_lift, rule_leverage, rule_conviction, covered_patterns_list]


def update_metrics(df):
    """
    Function that updates the metrics given a dataframe and a condition.
    Condition: precision < 0.5.

    :param df: dataframe with association rules and its metrics.
    :return new dataframe with updated metrics.
    """
    # Get every feature in a list.
    phenotype_list = [int(i) for i in df["Phenotype"].tolist()]
    rule_list = df["Rule"].tolist()
    precision_list = [float(i) for i in df["Precision"].tolist()]
    recall_list = [float(i) for i in df["Recall"].tolist()]
    lift_list = [float(i) for i in df["Lift"].tolist()]
    leverage_list = [float(i) for i in df["Leverage"].tolist()]
    conviction_list = [float(i) for i in df["Conviction"].tolist()]

    ant_support_list = [float(i) for i in df["S(A)"].tolist()]
    consec_support_list = [float(i) for i in df["S(C)"].tolist()]
    rule_support_list = [float(i) for i in df["S(A-->C)"].tolist()]

    covered_patterns_list = df["Covered patterns"].tolist()

    # Update consecuent.
    for i in range(len(precision_list)):
        if precision_list[i] < 0.5:  # Condition

            # Update consecuent.
            if rule_list[i].find("--> Si") != -1:
                rule_list[i] = rule_list[i].replace("--> Si", "--> No")
            elif rule_list[i].find("--> No") != -1:
                rule_list[i] = rule_list[i].replace("--> No", "--> Si")

            # Split rule: Antecedent / Consecuent.
            splitted_rule = rule_list[i].split(" --> ")

            # Get training and test data
            training_in, training_exp, test_in, test_exp = params[
                'FITNESS_FUNCTION'].training_in, params['FITNESS_FUNCTION'].training_exp, params['FITNESS_FUNCTION'].test_in, params['FITNESS_FUNCTION'].test_exp

            x = training_in
            y = training_exp

            # Get metrics.
            metrics = get_metrics(
                eval(splitted_rule[0]), y, splitted_rule[1], visualize=False)

            # Update old metrics.
            ant_support_list[i] = metrics[0]
            consec_support_list[i] = metrics[1]
            rule_support_list[i] = metrics[2]
            precision_list[i] = metrics[3]
            recall_list[i] = metrics[4]
            lift_list[i] = metrics[5]
            leverage_list[i] = metrics[6]
            conviction_list[i] = metrics[7]
            covered_patterns_list[i] = metrics[8]

    # Dataframes with the results.
    df_result = pd.DataFrame(list(zip(phenotype_list, rule_list, precision_list, recall_list, lift_list, leverage_list, conviction_list, ant_support_list, consec_support_list, rule_support_list, covered_patterns_list)),
                             columns=["Phenotype", "Rule", "Precision", "Recall", "Lift", "Leverage", "Conviction", "S(A)", "S(C)", "S(A-->C)", "Covered patterns"])

    return df_result


def get_min_avg_support(df):
    """
    Function that computes the minimum and the average support of a
    column dataframe.

    :param df: Column dataframe with support values.
    :return lhs: String with the minimum and average suppport as
    follows [min:avg].
        Example:    0.52:0.68
    """
    support_list = df.tolist()

    try:
        minimum_antec_support = min(support_list)
        avg_atec_support = sum(support_list) / \
                           len(support_list)
    except:
        minimum_antec_support= 0
        avg_atec_support = 0

    lhs = str(minimum_antec_support) + ":" + str(avg_atec_support)

    return lhs


def get_min_avg_confidence(df):
    """
    Function that computes the minimum and the average confidence of 
    a column dataframe.

    :param df: Column dataframe with confidence values.
    :return conf: String with the minimum and average confidence as
    follows [min:avg].

    Example:    0.52:0.68
    """
    confidence_list = df.tolist()

    minimum_conf = min(confidence_list)
    avg_conf = sum(confidence_list) / len(confidence_list)

    conf = str(minimum_conf) + ":" + str(avg_conf)

    return conf


def get_cond_in_antec(df):
    """
    Function that computes the amount of conditions within an
    antecedent given a rules column dataframe.

    :param df: Column dataframe with association rules.
    :return n_antecedents: Number of conditions within an antecedent.
    :return antec_counts: Number of rules that apply 'n_antecedents'.

    Example: 
        [1, 2, 3], [3, 2, 4]
        - There are 3 rules that have 1 condition.
        - There are 2 rules that have 2 conditions.
        - There are 4 rules that have 3 conditions.
    """
    n_conditions = []

    for rule in df:
        n_conditions.append(rule.count("&") + 1)


    max_conditions = np.max(n_conditions)

    n_antecedents = list(range(max_conditions + 1 ))

    antec_counts = [n_conditions.count(i) for i in n_antecedents]

    # n_antecedents, antec_counts = np.unique(n_conditions, return_counts=True)

    return n_antecedents, antec_counts


def get_attr_stats(df, n_features):
    """
    Function that compute both, the unique used attributes and 
    the distribution of the frequencies (%) of use of the attributes.

    :param df: Column dataframe with association rules.
    :param n_features: Amount of features in dataset.
    :return used_attr: Unique used attributes. Integer like.
    :return use_freq: Previous stats.
        Example:
            {Maximum, Q3, Median, Q2, Minimum} = {60, 15, 4, 1, 0}
    """
    # Create a list of zeros to count in how many rules each
    # attribute appears.
    features_list = [0] * n_features
    n_rules = len(df)

    attr = []
    for i in df:
        current_attrs = get_features_indexes(i)
        attr.append(current_attrs)

        for j in current_attrs:
            # Add +1 to the attribute if it appears in the current
            # rule. Since 'current_attrs' list has non-repeated values,
            # if a rule uses attribute 45 twice, it will be added just
            # once.
            features_list[j] += 1

    attr = [item for sublist in attr for item in sublist]
    flatten_attr = sorted(list(set(attr)))
    used_attr = len(flatten_attr)

    use_freq = ""
    if used_attr == 0:
        used_attr = "No rules obtained"
    else:
        # Attribute % use freq:
        attr_max = (max(features_list) / n_rules) * 100
        attr_q_3 = (np.quantile(features_list, .75) / n_rules) * 100
        attr_median = (statistics.median(features_list) / n_rules) * 100
        attr_q_2 = (np.quantile(features_list, .50) / n_rules) * 100
        attr_min = (min(features_list) / n_rules) * 100

        use_freq = "{" + str(attr_max) + "," + str(attr_q_3) + "," + \
            str(attr_median) + "," + str(attr_q_2) + "," + str(attr_min) + "}"

    return used_attr, use_freq


def get_uncovered_patterns(df):
    """
    Function that computes the % uncovered patterns given a column
    dataframe of association rules.

    :param df: Column dataframe with association rules.
    :return not_covered_positives: Percentage of uncovered positive patterns.
    """
    # Get the covered patterns by each rule.
    lists = df.tolist()

    # Get all patterns in a flat list.
    result = [int(item) for sublist in lists for item in sublist]

    # Make them unique.
    covered_patterns_list = sorted(list(set(result)))

    # Get targets.
    y = params['FITNESS_FUNCTION'].training_exp.to_frame()

    # Drop not positive targets.
    y = y.drop(y[y["COMPLICACIONES"] == "No"].index)

    positive_patterns_list = y.index.tolist()

    # Amount of covered patterns by the association rules.
    covered_patterns = len(covered_patterns_list)

    # Amount of positive patterns.
    positive_patterns = len(positive_patterns_list)

    # Amount of positive and covered patterns.
    covered_and_positive = 0
    for i in covered_patterns_list:
        if i in positive_patterns_list:
            covered_and_positive += 1

    not_covered_positives = (
        (covered_patterns - covered_and_positive) / covered_patterns) * 100

    return not_covered_positives
