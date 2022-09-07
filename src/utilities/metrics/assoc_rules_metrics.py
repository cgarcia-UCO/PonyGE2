import sys

import numpy as np
import re

class AssocRules_Stats:

    def __init__(self):
        pass

    def compute_support(self, rules, x):
        values = []

        num_rows = x.shape[0]

        for i in rules:
            values.append(sum(eval(i.get_antecedent()))/num_rows)

        return values

    def compute_confidence(self, rules, x, y):
        values = []

        for i in rules:
            covered_patterns = eval(i.get_antecedent())
            true_class = y[covered_patterns]
            predicted_class = i.get_consequent().replace('\'','')
            hits = true_class == predicted_class
            try:
                values.append(sum(hits) / sum(covered_patterns))
            except ZeroDivisionError:
                values.append(0)

        return values

    def get_lengths(self, rules):
        values = []
        num_rules_processed = 0
        iterator = 0

        while num_rules_processed < len(rules):
            new_rules = [i for i in rules if len(i.conditions) == iterator]
            num_new_rules = len(new_rules)
            values.append(num_new_rules)
            num_rules_processed += num_new_rules
            iterator += 1

        return values

    def get_used_attributes(self, rules):
        attributes = set()
        regex_attribute = "\[\'([^\]]+)\'\]"

        for i in rules:
            for j in i.conditions:
                attribute = re.search(regex_attribute, j).group(1)
                attributes.add(attribute)

        return attributes

    def get_freq_attributes(self, rules, dataset):
        values = []

        for i in dataset.columns:
            # Number of rules with a condition involving the dataset feature i
            frequency = len([i_rules for i_rules in rules if
                             len([j for j in i_rules.conditions if i in j]) > 0])
            values.append(frequency / len(rules))

        return np.array(values)

    def get_uncovered(self, rules, x, y, target_class):
        index_target_patterns = y == target_class

        for i in rules:
            covered_patterns = eval(i.get_antecedent())
            index_target_patterns = index_target_patterns & (~ covered_patterns)

        return x[index_target_patterns]

    def print_stats(self, rules, x, y, f = sys.stdout):
        try:
            support_values = self.compute_support(rules, x)
            min_support = min(support_values)
            avg_support = np.mean(support_values)

            confidence_values = self.compute_confidence(rules, x, y)
            min_confidence = min(confidence_values)

            assert min_confidence >= 0.5 # TODO quitar al final, por si no se usa prune_if_else_tree

            avg_confidece = np.mean(confidence_values)
            # max_confidence = max(confidence_values)

            num_rules_per_length = self.get_lengths(rules)

            used_attributes = self.get_used_attributes(rules)

            freq_attributes = self.get_freq_attributes(rules, x) * 100

            uncovered_patterns = self.get_uncovered(rules, x, y, 'Si')

            f.write("{:.1f}".format(min_support*100)+":"+"{:.1f}".format(avg_support*100)+'\t')

            f.write("{:.1f}".format(min_confidence*100)+":"+"{:.1f}".format(avg_confidece*100)+'\t')

            f.write(str(len(rules))+'\t')

            f.write(str(num_rules_per_length)+'\t')

            f.write(str(len(used_attributes))+'\t')

            f.write('{'+"{:.0f}".format(max(freq_attributes))+','+
                  "{:.0f}".format(np.percentile(freq_attributes, 75))+','+
                  "{:.0f}".format(np.median(freq_attributes))+','+
                  "{:.0f}".format(np.percentile(freq_attributes, 25))+','+
                  "{:.0f}".format(min(freq_attributes))+'}'+'\t')

            f.write("{:.0f}".format(len(uncovered_patterns)/len(y)*100))
        except:
            if len(rules) > 0:
                f.write('Error')
                raise
            else:
                f.write('No rules')
        f.write("\n")
