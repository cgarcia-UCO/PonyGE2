import sys
from math import floor
from re import DOTALL, MULTILINE, finditer, match
from sys import maxsize

import numpy as np
import pandas as pd

from algorithm.parameters import params
from utilities.misc.readappend_StringIO import ReadAppend_StringIO
from utilities.misc.get_subsets import get_subsets


class Grammar(object):
    """
    Parser for Backus-Naur Form (BNF) Context-Free Grammars.
    """

    def __init__(self, file_name):
        """
        Initialises an instance of the grammar class. This instance is used
        to parse a given file_name grammar.

        :param file_name: A specified BNF grammar file.
        """

        if file_name.endswith("pybnf"):
            # Use python filter for parsing grammar output as grammar output
            # contains indented python code.
            self.python_mode = True

        else:
            # No need to filter/interpret grammar output, individual
            # phenotypes can be evaluated as normal.
            self.python_mode = False

        # Initialise empty dict for all production rules in the grammar.
        # Initialise empty dict of permutations of solutions possible at
        # each derivation tree depth.
        self.rules, self.permutations = {}, {}

        # Initialise dicts for terminals and non terminals, set params.
        self.non_terminals, self.terminals = {}, {}
        self.start_rule, self.codon_size = None, params['CODON_SIZE']
        self.min_path, self.max_arity, self.min_ramp = None, None, None

        # Set regular expressions for parsing BNF grammar.
        self.ruleregex = '(?P<rulename><\S+>)\s*::=\s*(?P<production>(?:(?=\#)\#[^\r\n]*|(?!<\S+>\s*::=).+?)+)'
        self.productionregex = '(?=\#)(?:\#.*$)|(?!\#)\s*(?P<production>(?:[^\'\"\|\#]+|\'.*?\'|".*?")+)'
        self.productionpartsregex = '\ *([\r\n]+)\ *|([^\'"<\r\n]+)|\'(.*?)\'|"(.*?)"|(?P<subrule><[^>|\s]+>)|([<]+)'

        # Read in BNF grammar, set production rules, terminals and
        # non-terminals.
        self.read_bnf_file(file_name)

        # Check the minimum depths of all non-terminals in the grammar.
        self.check_depths()

        # Check which non-terminals are recursive.
        self.check_recursion(self.start_rule["symbol"], [])

        # Set the minimum path and maximum arity of the grammar.
        self.set_arity()

        # Generate lists of recursive production choices and shortest
        # terminating path production choices for each NT in the grammar.
        # Enables faster tree operations.
        self.set_grammar_properties()

        # Calculate the total number of derivation tree permutations and
        # combinations that can be created by a grammar at a range of depths.
        self.check_permutations()

        if params['MIN_INIT_TREE_DEPTH']:
            # Set the minimum ramping tree depth from the command line.
            self.min_ramp = params['MIN_INIT_TREE_DEPTH']

        elif hasattr(params['INITIALISATION'], "ramping"):
            # Set the minimum depth at which ramping can start where we can
            # have unique solutions (no duplicates).
            self.get_min_ramp_depth()

        if params['REVERSE_MAPPING_TARGET'] or params['TARGET_SEED_FOLDER']:
            # Initialise dicts for reverse-mapping GE individuals.
            self.concat_NTs, self.climb_NTs = {}, {}

            # Find production choices which can be used to concatenate
            # subtrees.
            self.find_concatenation_NTs()

    def read_bnf_file(self, file_name):
        """
        Read a grammar file in BNF format. Uses read_bnf_stringio after
        reading the file into a StringIO object

        :param file_name: A specified BNF grammar file
        :return: Nothing.
        """

        with open(file_name, 'r') as bnf:
            self.read_bnf_stringio(ReadAppend_StringIO(bnf.read()))

    def read_bnf_stringio(self, grammar_content):
        """
        Read a grammar, from a StringIO, in BNF format. Parses the grammar and saves a
        dict of all production rules and their possible choices.

        :param file_name: A specified BNF grammar file (already open or in a StringIO buffer).
        :return: Nothing.
        """
        bnf = grammar_content

        # Read the whole grammar file. This is now in a while because the ReadAppend_StringIO can be extended (append)
        # while being read. My intention is that, in case of reading special symbols, extend on-the-fly the grammar
        # with automatically generated rules and rule-productions
        content = bnf.read()
        while len(content) > 0:

            for rule in finditer(self.ruleregex, content, DOTALL):
                # Find all rules in the grammar

                if self.start_rule is None:
                    # Set the first rule found as the start rule.
                    self.start_rule = {"symbol": rule.group('rulename'),
                                       "type": "NT"}

                # Create and add a new rule.
                self.non_terminals[rule.group('rulename')] = {
                    'id': rule.group('rulename'),
                    'min_steps': maxsize,
                    'expanded': False,
                    'recursive': True,
                    'b_factor': 0}

                # Initialise empty list of all production choices for this
                # rule.
                tmp_productions = []

                for p in finditer(self.productionregex,
                                  rule.group('production'), MULTILINE):
                    # Iterate over all production choices for this rule.
                    # Split production choices of a rule.

                    if p.group('production') is None or p.group(
                            'production').isspace():
                        # Skip to the next iteration of the loop if the
                        # current "p" production is None or blank space.
                        continue

                    # Initialise empty data structures for production choice
                    tmp_production, terminalparts = [], None

                    # Try processing GE_GENERATE: cases
                    # In case the processing was erroneous, for instance because of the presence of
                    # a tag incompatible with the dataset, skip the tag
                    if self.try_processing_generate_cases(p, grammar_content) == 'not valid tag':
                        continue

                    # special cases: GE_RANGE:dataset_n_vars will be
                    # transformed to productions 0 | 1 | ... |
                    # n_vars-1, and similar for dataset_n_is,
                    # dataset_n_os
                    GE_RANGE_regex = r'GE_RANGE:(?P<range>\w*)'
                    m = match(GE_RANGE_regex, p.group('production'))
                    if m:
                        try:
                            if m.group('range') == "dataset_n_vars":
                                # number of columns from dataset
                                n = params['FITNESS_FUNCTION'].n_vars
                            elif m.group('range') == "dataset_n_is":
                                # number of input symbols (see
                                # if_else_classifier.py)
                                n = params['FITNESS_FUNCTION'].n_is
                            elif m.group('range') == "dataset_n_os":
                                # number of output symbols
                                n = params['FITNESS_FUNCTION'].n_os
                            else:
                                # assume it's just an int
                                n = int(m.group('range'))
                        except (ValueError, AttributeError):
                            raise ValueError("Bad use of GE_RANGE: "
                                             + m.group())

                        for i in range(n):
                            # add a terminal symbol
                            tmp_production, terminalparts = [], None
                            symbol = {
                                "symbol": str(i),
                                "type": "T",
                                "min_steps": 0,
                                "recursive": False}
                            tmp_production.append(symbol)
                            if str(i) not in self.terminals:
                                self.terminals[str(i)] = \
                                    [rule.group('rulename')]
                            elif rule.group('rulename') not in \
                                    self.terminals[str(i)]:
                                self.terminals[str(i)].append(
                                    rule.group('rulename'))
                            tmp_productions.append({"choice": tmp_production,
                                                    "recursive": False,
                                                    "NT_kids": False})
                        # don't try to process this production further
                        # (but later productions in same rule will work)
                        continue

                    for sub_p in finditer(self.productionpartsregex,
                                          p.group('production').strip()):
                        # Split production into terminal and non terminal
                        # symbols.

                        if sub_p.group('subrule'):
                            if terminalparts is not None:
                                # Terminal symbol is to be appended to the
                                # terminals dictionary.
                                symbol = {"symbol": terminalparts,
                                          "type": "T",
                                          "min_steps": 0,
                                          "recursive": False}
                                tmp_production.append(symbol)
                                if terminalparts not in self.terminals:
                                    self.terminals[terminalparts] = \
                                        [rule.group('rulename')]
                                elif rule.group('rulename') not in \
                                        self.terminals[terminalparts]:
                                    self.terminals[terminalparts].append(
                                        rule.group('rulename'))
                                terminalparts = None

                            tmp_production.append(
                                {"symbol": sub_p.group('subrule'),
                                 "type": "NT"})

                        else:
                            # Unescape special characters (\n, \t etc.)
                            if terminalparts is None:
                                terminalparts = ''
                            terminalparts += ''.join(
                                [part.encode().decode('unicode-escape') for
                                 part in sub_p.groups() if part])

                    if terminalparts is not None:
                        # Terminal symbol is to be appended to the terminals
                        # dictionary.
                        symbol = {"symbol": terminalparts,
                                  "type": "T",
                                  "min_steps": 0,
                                  "recursive": False}
                        tmp_production.append(symbol)
                        if terminalparts not in self.terminals:
                            self.terminals[terminalparts] = \
                                [rule.group('rulename')]
                        elif rule.group('rulename') not in \
                                self.terminals[terminalparts]:
                            self.terminals[terminalparts].append(
                                rule.group('rulename'))
                    tmp_productions.append({"choice": tmp_production,
                                            "recursive": False,
                                            "NT_kids": False})

                assert len(tmp_productions) <= self.codon_size, 'There is a rule with ' + str(len(tmp_productions)) + \
                       ' productions, when the maximum is ' + str(self.codon_size) + \
                       '. You may want to specify a larger CODON_SIZE parameter value.'

                if not rule.group('rulename') in self.rules:
                    # Add new production rule to the rules dictionary if not
                    # already there.
                    self.rules[rule.group('rulename')] = {
                        "choices": tmp_productions,
                        "no_choices": len(tmp_productions)}

                    if len(tmp_productions) == 1:
                        # Unit productions.
                        print("Warning: Grammar contains unit production "
                              "for production rule", rule.group('rulename'))
                        print("         Unit productions consume GE codons.")
                else:
                    # Conflicting rules with the same name.
                    raise ValueError("lhs should be unique",
                                     rule.group('rulename'))

            content = bnf.read()

        if params['DEBUG']:
            print(grammar_content.getvalue())

    def _generate_subsets(self, i, grammar_content, min_size=2, max_size=8):
        """
        Extend the grammar with production rules with all the subset values for the i-th feature,
        derived from the values in the dataset of the fitness function,
        which is expected to be an instance of supervised_learning.supervised_learning

        :param i: Index of the feature of the dataset considered for generating the new production rule
        :param grammar_content: StringIO buffer object where new rules are appended
        :return: Nothing.
        """

        # In case this tag was not previously used
        if self.ge_generate_tags.get('subset_values_feature_' + str(i), 'not used') == 'not used':
            self.ge_generate_tags['subset_values_feature_' + str(i)] = 'used'

            inputs = params['FITNESS_FUNCTION'].training_in

            if isinstance(inputs, pd.DataFrame):
                values = inputs.iloc[:,i]
                values = values[values.notna()]
                values = np.unique(values)
            elif isinstance(inputs, np.ndarray):
                values = np.unique(inputs[:, i])
            else:
                raise Exception('Training dataset is not a Numpy.ndarray'
                                ' nor a pandas.DataFrame: ' + type(self.training_in))

            values = list(get_subsets(list(values), min_size=min_size,
                                      max_size=max_size))
            header_required_quotation = '\"'
            tail_required_quotation = '\"'

            grammar_content.append('<subset_values_feature_' + str(i) + '> ::= ' + header_required_quotation
                                   + str(values[0]) + tail_required_quotation)

            for j in values[1:]:
                grammar_content.append(' | ' + header_required_quotation + str(j) + tail_required_quotation)

            grammar_content.append('\n')

    def _generate_values_feature(self, i, grammar_content):
        """
        Extend the grammar with production rules for the values of the i-th feature,
        derived from the values in the dataset of the fitness function,
        which is expected to be an instance of supervised_learning.supervised_learning

        :param i: Index of the feature of the dataset considered for generating the new production rules
        :param grammar_content: StringIO buffer object where new rules are appended
        :return: Nothing.
        """

        # In case this tag was not previously used
        if self.ge_generate_tags.get('values_feature_' + str(i), 'not used') == 'not used':
            self.ge_generate_tags['values_feature_' + str(i)] = 'used'

            inputs = params['FITNESS_FUNCTION'].training_in

            if isinstance(inputs, pd.DataFrame):
                values = inputs.iloc[:,i]
                values = values[values.notna()]
                values = np.unique(values)
            elif isinstance(inputs, np.ndarray):
                values = np.unique(inputs[:, i])
            else:
                raise Exception('Training dataset is not a Numpy.ndarray'
                                ' nor a pandas.DataFrame: ' + type(self.training_in))

            header_required_quotation = '\"\'' if params['FITNESS_FUNCTION'].is_ithfeature_categorical(i) else ''
            tail_required_quotation = '\'\"' if params['FITNESS_FUNCTION'].is_ithfeature_categorical(i) else ''

            grammar_content.append('<values_feature_' + str(i) + '> ::= ' + header_required_quotation
                                   + str(values[0]) + tail_required_quotation)

            for j in values[1:]:
                grammar_content.append(' | ' + header_required_quotation + str(j) + tail_required_quotation)

            grammar_content.append('\n')

    def try_processing_generate_cases(self, p, grammar_content):
        """
        Extend the grammar with production rules derived from the values in the dataset of the fitness function,
        which is expected to be an instance of supervised_learning.supervised_learning.

        Some types of production rules  are available just for categorical features, other just for numerical features.
        A feature is considered categorical in case the dataset contains no more than 10 different values.
        Otherwise, it is considered numerical.

        :param p: Production of the grammar being currently processed
        :param grammar_content: StringIO buffer object where new rules are appended
        :return: 'not valid tag', in case the tag is not compatible with the dataset, 'ok' otherwise.
        """

        # special cases: GE_GENERATE:****
        # constructions will go over the dataset, creating possible conditions according to the values in it
        GE_GENERATE_regex = r'<GE_GENERATE:(?P<type_generation>\w*)>'
        num_generate_tags_in_production = 0
        for m in finditer(GE_GENERATE_regex,
                          p.group('production'), MULTILINE):

            num_generate_tags_in_production += 1

            if num_generate_tags_in_production > 1:
                print("Error: Production with more than one GE_GENERATE tags. This functionality is not yet available:",
                      p.group('production'))
                sys.exit(1)

            # Some production rules must not be generated twice. Thus, this new member variable records those
            # set of production produced.
            if not hasattr(self, 'ge_generate_tags'):
                self.ge_generate_tags = {}

            try:
                not_valid_tag = 'not valid tag'
                # Generation of not equity conditions for the categorical features in the dataset
                if m.group('type_generation') == "dataset_neq_conditions":
                    if 0 == self._generate_neq_conditions_rules(grammar_content):
                        return not_valid_tag

                # Generation of equity conditions for the features in the dataset
                elif m.group('type_generation') == "dataset_eq_conditions":
                    if 0 == self._generate_eq_conditions_rules(grammar_content):
                        return not_valid_tag

                # Generation of target labels
                elif m.group('type_generation') == "dataset_target_labels":
                    self._generate_target_labels(grammar_content)

                elif m.group('type_generation') == 'dataset_lessequal_conditions':
                    if 0 == self._generate_lessequal_condition_rules(grammar_content):
                        return not_valid_tag

                elif m.group('type_generation') == 'dataset_inset_conditions':
                    if 0 == self._generate_inset_conditions_rules(grammar_content):
                        return not_valid_tag

                elif m.group('type_generation') == 'dataset_notin_conditions':
                    if 0 == self._generate_notin_conditions_rules(grammar_content):
                        return not_valid_tag

                elif m.group('type_generation') == 'dataset_greater_conditions':
                    if 0 == self._generate_greater_conditions_rules(grammar_content):
                        return not_valid_tag

                elif m.group('type_generation') == 'dataset_numeric_labels':
                    if 0 == self._generate_numeric_labels(grammar_content):
                        return not_valid_tag

                else:
                    raise ValueError("Bad use of GE_GENERATE:" + m.group())

            except (ValueError, AttributeError):
                raise ValueError("Bad use of GE_GENERATE: "
                                 + m.group())

        return 'ok'

    def _generate_numeric_labels(self, grammar_content):
        """
        Extend the grammar with a production rule for the indexes of numeric features in the dataset,
        which is expected to be an instance of supervised_learning.classification.

        :param grammar_content: StringIO buffer object where a new rule is appended
        :return: The number of found numeric features.
        """
        if self.ge_generate_tags.get('dataset_numeric_labels', 'not used') == 'not used':
            self.ge_generate_tags['dataset_numeric_labels'] = 'used'
            # append a new rule in the grammar with the shape
            # <GE_GENERATE:dataset_numeric_labels> ::= index_1 | index_2 | ...
            num_processed_features = 0
            first_feature = params['FITNESS_FUNCTION'].get_first_numerical_feature()
            inputs = params['FITNESS_FUNCTION'].training_in

            assert isinstance(inputs, pd.DataFrame), "Tag GE_GENERATE:dataset_numeric_labels is only available" \
                                                     " for pandas Dataframe, not Numpy ndarray"

            if first_feature is not None:
                grammar_content.append('\n<GE_GENERATE:dataset_numeric_labels> ::= ')
                num_processed_features += 1
                # Go over the features of the dataset.
                grammar_content.append('\"\'' + inputs.columns[first_feature] + '\'\"')
                for i in range(first_feature + 1, inputs.shape[1]):
                    if not params['FITNESS_FUNCTION'].is_ithfeature_categorical(i):
                        grammar_content.append(' | \"\'' + inputs.columns[i] + '\'\"')
                        num_processed_features += 1

                grammar_content.append('\n')

            self.ge_generate_tags['found_numeric_indexes'] = num_processed_features

        return self.ge_generate_tags['found_numeric_indexes']

    def _generate_target_labels(self, grammar_content):
        """
        Extend the grammar with a production rule with the values of different values of the target output,
        derived from the values in the dataset of the fitness function,
        which is expected to be an instance of supervised_learning.classification.

        :param grammar_content: StringIO buffer object where new rules are appended
        :return: Nothing.
        """

        # In case this tag was not previously used
        if self.ge_generate_tags.get('dataset_target_labels', 'not used') == 'not used':
            self.ge_generate_tags['dataset_target_labels'] = 'used'
            # append a new rule in the grammar with the shape
            # <GE_GENERATE:dataset_target_labels> ::= label_1 | label_2 | ...
            labels = np.unique(params['FITNESS_FUNCTION'].training_exp)
            header_required_quotation = '' if issubclass(labels.dtype.type, np.number) else '\"\''
            tail_required_quotation = '' if issubclass(labels.dtype.type, np.number) else '\'\"'
            grammar_content.append('\n<GE_GENERATE:dataset_target_labels> ::= ' + header_required_quotation
                                   + str(labels[0]) + tail_required_quotation)
            for j in labels[1:]:
                grammar_content.append(' | ' + header_required_quotation + str(j) + tail_required_quotation)
            grammar_content.append('\n')

    def _generate_notin_conditions_rules(self, grammar_content):
        """
        Extend the grammar with production rules with not in conditions,
        derived from the values in the dataset of the fitness function,
        which is expected to be an instance of supervised_learning.supervised_learning.

        This type of new production rules are generated just for categorical features

        :param grammar_content: StringIO buffer object where new rules are appended
        :return: The number of processed features (this time or ever).
        """
        # In case this tag was not previously used
        if self.ge_generate_tags.get('dataset_notin_conditions', 'not used') == 'not used':
            self.ge_generate_tags['dataset_notin_conditions'] = 'used'
            # append a new rule in the grammar with the shape
            # <GE_GENERATE:dataset_inset_conditions> ::= x[0] in <value_0> | x[1] in <value_1> | ...
            # and others for each <value_i> with the shape:
            # <value_i> ::= <<first unique subset of values in x[0]>> | <<second unique...
            num_processed_features = 0
            inputs = params['FITNESS_FUNCTION'].training_in
            first_feature = params['FITNESS_FUNCTION'].get_first_categorical_feature()

            if first_feature is not None:
                while first_feature < inputs.shape[1] and\
                        (params['FITNESS_FUNCTION'].num_of_different_values(first_feature) <= 2 or
                         not params['FITNESS_FUNCTION'].is_ithfeature_categorical(first_feature)):
                    first_feature += 1

                if first_feature < inputs.shape[1]:
                    grammar_content.append('\n<GE_GENERATE:dataset_notin_conditions> ::= ')
                    # Go over the features of the dataset.
                    # This code assumes params['FITNESS_FUNCTION'] is a supervised_learning.supervised_learning object
                    header = '(~np.isin(x[\"\'' if isinstance(inputs, pd.DataFrame) else '(~np.isin(x[:,'
                    tail = '\'\"]' if isinstance(inputs, pd.DataFrame) else ']'
                    grammar_content.append(header + inputs.columns[first_feature] + tail + ', <subset_values_feature_' + str(first_feature) + '>))')
                    for i in range(first_feature + 1, inputs.shape[1]):
                        if params['FITNESS_FUNCTION'].is_ithfeature_categorical(i) and\
                                params['FITNESS_FUNCTION'].num_of_different_values(i) > 2:
                            grammar_content.append(' | '+ header + inputs.columns[i] + tail + ', <subset_values_feature_' + str(i) + '>))')

                    grammar_content.append('\n')

                    for i in range(first_feature, inputs.shape[1]):
                        if params['FITNESS_FUNCTION'].is_ithfeature_categorical(i) and\
                                params['FITNESS_FUNCTION'].num_of_different_values(i) > 2:
                            self._generate_subsets(i, grammar_content)
                            num_processed_features += 1

            self.ge_generate_tags['notin_processed_features'] = num_processed_features

        return self.ge_generate_tags['notin_processed_features']

    def _generate_inset_conditions_rules(self, grammar_content):
        """
        Extend the grammar with production rules with inset conditions,
        derived from the values in the dataset of the fitness function,
        which is expected to be an instance of supervised_learning.supervised_learning.

        This type of new production rules are generated just for categorical features

        :param grammar_content: StringIO buffer object where new rules are appended
        :return: The number of processed features (this time or ever).
        """
        # In case this tag was not previously used
        if self.ge_generate_tags.get('dataset_inset_conditions', 'not used') == 'not used':
            self.ge_generate_tags['dataset_inset_conditions'] = 'used'
            # append a new rule in the grammar with the shape
            # <GE_GENERATE:dataset_inset_conditions> ::= x[0] in <value_0> | x[1] in <value_1> | ...
            # and others for each <value_i> with the shape:
            # <value_i> ::= <<first unique subset of values in x[0]>> | <<second unique...
            num_processed_features = 0
            inputs = params['FITNESS_FUNCTION'].training_in
            first_feature = params['FITNESS_FUNCTION'].get_first_categorical_feature()

            if first_feature is not None:
                while first_feature < inputs.shape[1] and\
                        (params['FITNESS_FUNCTION'].num_of_different_values(first_feature) <= 2 or
                         not params['FITNESS_FUNCTION'].is_ithfeature_categorical(first_feature)):
                    first_feature += 1

                if first_feature < inputs.shape[1]:
                    grammar_content.append('\n<GE_GENERATE:dataset_inset_conditions> ::= ')
                    # Go over the features of the dataset.
                    # This code assumes params['FITNESS_FUNCTION'] is a supervised_learning.supervised_learning object
                    header = '(np.isin(x[\"\'' if isinstance(inputs, pd.DataFrame) else 'np.isin(x[:,'
                    tail = '\'\"]' if isinstance(inputs, pd.DataFrame) else ']'
                    grammar_content.append(header + inputs.columns[first_feature] + tail + ', <subset_values_feature_' + str(first_feature) + '>))')
                    for i in range(first_feature + 1, inputs.shape[1]):
                        if params['FITNESS_FUNCTION'].is_ithfeature_categorical(i) and\
                                params['FITNESS_FUNCTION'].num_of_different_values(i) > 2:
                            grammar_content.append(' | '+ header + inputs.columns[i] + tail + ', <subset_values_feature_' + str(i) + '>))')

                    grammar_content.append('\n')

                    for i in range(first_feature, inputs.shape[1]):
                        if params['FITNESS_FUNCTION'].is_ithfeature_categorical(i) and\
                                params['FITNESS_FUNCTION'].num_of_different_values(i) > 2:
                            self._generate_subsets(i, grammar_content)
                            num_processed_features += 1

            self.ge_generate_tags['inset_processed_features'] = num_processed_features

        return self.ge_generate_tags['inset_processed_features']

    def _generate_neq_conditions_rules(self, grammar_content):
        """
        Extend the grammar with production rules with NOT equity conditions,
        derived from the values in the dataset of the fitness function,
        which is expected to be an instance of supervised_learning.supervised_learning.

        This type of new production rules are generated just for categorical features

        :param grammar_content: StringIO buffer object where new rules are appended
        :return: The number of processed features (this time or ever).
        """

        # In case this tag was not previously used
        if self.ge_generate_tags.get('dataset_neq_conditions', 'not used') == 'not used':
            self.ge_generate_tags['dataset_neq_conditions'] = 'used'
            # append a new rule in the grammar with the shape
            # <GE_GENERATE:dataset_eq_conditions> ::= x[0] != <value_0> | x[1] != <value_1>
            # and others for each <value_i> with the shape:
            # <value_i> ::= <<first unique value in x[0]>> | <<second unique...
            num_processed_features = 0
            first_feature = params['FITNESS_FUNCTION'].get_first_categorical_feature()
            inputs = params['FITNESS_FUNCTION'].training_in
            if first_feature is not None:
                while first_feature < inputs.shape[1] and \
                        (params['FITNESS_FUNCTION'].num_of_different_values(first_feature) <= 1 or
                         not params['FITNESS_FUNCTION'].is_ithfeature_categorical(first_feature)):
                    first_feature += 1

                if first_feature < inputs.shape[1]:
                    grammar_content.append('\n<GE_GENERATE:dataset_neq_conditions> ::= ')
                    # Go over the features of the dataset.
                    # This code assumes params['FITNESS_FUNCTION'] is a supervised_learning.supervised_learning object
                    header = '(x[\"\'' if isinstance(inputs, pd.DataFrame) else '(x[:,'
                    tail = '\'\"]' if isinstance(inputs, pd.DataFrame) else ']'
                    grammar_content.append(header + inputs.columns[first_feature] + tail + ' != <values_feature_' + str(first_feature) + '>)')
                    for i in range(first_feature + 1, inputs.shape[1]):
                        if params['FITNESS_FUNCTION'].is_ithfeature_categorical(i) and\
                                params['FITNESS_FUNCTION'].num_of_different_values(i) > 1:
                            grammar_content.append(' | '+ header + inputs.columns[i] + tail + ' != <values_feature_' + str(i) + '>)')

                    grammar_content.append('\n')

                    for i in range(first_feature, inputs.shape[1]):
                        if params['FITNESS_FUNCTION'].is_ithfeature_categorical(i) and\
                                params['FITNESS_FUNCTION'].num_of_different_values(i) > 1:
                            self._generate_values_feature(i, grammar_content)
                            num_processed_features += 1

            self.ge_generate_tags['neq_processed_features'] = num_processed_features

        return self.ge_generate_tags['neq_processed_features']

    def _generate_eq_conditions_rules(self, grammar_content):
        """
        Extend the grammar with production rules with equity conditions,
        derived from the values in the dataset of the fitness function,
        which is expected to be an instance of supervised_learning.supervised_learning.

        This type of new production rules are generated just for categorical features

        :param grammar_content: StringIO buffer object where new rules are appended
        :return: The number of processed features (this time or ever).
        """

        # In case this tag was not previously used
        if self.ge_generate_tags.get('dataset_eq_conditions', 'not used') == 'not used':
            self.ge_generate_tags['dataset_eq_conditions'] = 'used'
            # append a new rule in the grammar with the shape
            # <GE_GENERATE:dataset_eq_conditions> ::= x[0] == <value_0> | x[1] == <value_1>
            # and others for each <value_i> with the shape:
            # <value_i> ::= <<first unique value in x[0]>> | <<second unique...
            num_processed_features = 0
            first_feature = params['FITNESS_FUNCTION'].get_first_categorical_feature()

            if params['EXPERIMENT_NAME'] and params['EXPERIMENT_NAME'].startswith('ponyge2'): #TODO eliminar
                first_feature = 0

            inputs = params['FITNESS_FUNCTION'].training_in
            if first_feature is not None:
                # TODO eliminar la siguiente primera condición params['EXPERIMENT_NAME'] != 'ponyge2'
                while (not params['EXPERIMENT_NAME'].startswith('ponyge2')) and \
                        first_feature < inputs.shape[1] and \
                        (params['FITNESS_FUNCTION'].num_of_different_values(first_feature) <= 1 or
                         not params['FITNESS_FUNCTION'].is_ithfeature_categorical(first_feature)):
                    first_feature += 1

                if first_feature < inputs.shape[1]:
                    grammar_content.append('\n<GE_GENERATE:dataset_eq_conditions> ::= ')
                    # Go over the features of the dataset.
                    # This code assumes params['FITNESS_FUNCTION'] is a supervised_learning.supervised_learning object
                    header = '(x[\"\'' if isinstance(inputs, pd.DataFrame) else '(x[:,'
                    tail = '\'\"]' if isinstance(inputs, pd.DataFrame) else ']'
                    grammar_content.append(header + inputs.columns[first_feature] + tail + ' == <values_feature_' + str(first_feature) + '>)')
                    for i in range(first_feature + 1, inputs.shape[1]):
                        # TODO eliminar la siguiente primera condición params['EXPERIMENT_NAME'] == 'ponyge2'
                        if params['EXPERIMENT_NAME'].startswith('ponyge2') or (params['FITNESS_FUNCTION'].is_ithfeature_categorical(i) and\
                                params['FITNESS_FUNCTION'].num_of_different_values(i) > 1):
                            grammar_content.append(' | ' + header + inputs.columns[i] + tail + ' == <values_feature_' + str(i) + '>)')

                    grammar_content.append('\n')

                    for i in range(first_feature, inputs.shape[1]):
                        # TODO eliminar la siguiente primera condición params['EXPERIMENT_NAME'] == 'ponyge2'
                        if params['EXPERIMENT_NAME'].startswith('ponyge2') or (params['FITNESS_FUNCTION'].is_ithfeature_categorical(i) and\
                                params['FITNESS_FUNCTION'].num_of_different_values(i) > 1):
                            self._generate_values_feature(i, grammar_content)
                            num_processed_features += 1

            self.ge_generate_tags['eq_processed_features'] = num_processed_features

        return self.ge_generate_tags['eq_processed_features']

    def _generate_greater_conditions_rules(self, grammar_content):
        """
        Extend the grammar with production rules with greater than conditions,
        derived from the values in the dataset of the fitness function,
        which is expected to be an instance of supervised_learning.supervised_learning.

        This type of new production rules are generated just for numerical features

        :param grammar_content: StringIO buffer object where new rules are appended
        :return: The number of processed features (this time or ever).
        """

        # In case this tag was not previously used
        if self.ge_generate_tags.get('dataset_greater_conditions', 'not used') == 'not used':
            self.ge_generate_tags['dataset_greater_conditions'] = 'used'
            # append a new rule in the grammar with the shape
            # <GE_GENERATE:dataset_eq_conditions> ::= x[0] <= <value_0> | x[1] <= <value_1>
            # and others for each <value_i> with the shape:
            # <value_i> ::= <<first unique value in x[0]>> | <<second unique...
            num_processed_features = 0
            first_feature = params['FITNESS_FUNCTION'].get_first_numerical_feature()
            inputs = params['FITNESS_FUNCTION'].training_in
            if first_feature is not None:
                grammar_content.append('\n<GE_GENERATE:dataset_greater_conditions> ::= ')
                # Go over the features of the dataset.
                # This code assumes params['FITNESS_FUNCTION'] is a supervised_learning.supervised_learning object
                header = '(x[\"\'' if isinstance(inputs, pd.DataFrame) else '(x[:,'
                tail = '\'\"]' if isinstance(inputs, pd.DataFrame) else ']'
                grammar_content.append(header + inputs.columns[first_feature] + tail + ' > <values_feature_' + str(first_feature) + '>)')
                for i in range(first_feature + 1, inputs.shape[1]):
                    if not params['FITNESS_FUNCTION'].is_ithfeature_categorical(i):
                        grammar_content.append(' | ' + header + inputs.columns[i] + tail + ' > <values_feature_' + str(i) + '>)')

                grammar_content.append('\n')

                for i in range(first_feature, inputs.shape[1]):
                    if not params['FITNESS_FUNCTION'].is_ithfeature_categorical(i):
                        self._generate_values_feature(i, grammar_content)
                        num_processed_features += 1

            self.ge_generate_tags['greater_processed_features'] = num_processed_features

        return self.ge_generate_tags['greater_processed_features']

    def _generate_lessequal_condition_rules(self, grammar_content):
        """
        Extend the grammar with production rules with less than or equal to conditions,
        derived from the values in the dataset of the fitness function,
        which is expected to be an instance of supervised_learning.supervised_learning.

        This type of new production rules are generated just for numerical features

        :param grammar_content: StringIO buffer object where new rules are appended
        :return: The number of processed features (this time or ever).
        """

        # In case this tag was not previously used
        if self.ge_generate_tags.get('dataset_lessequal_conditions', 'not used') == 'not used':
            self.ge_generate_tags['dataset_lessequal_conditions'] = 'used'
            # append a new rule in the grammar with the shape
            # <GE_GENERATE:dataset_eq_conditions> ::= x[0] <= <value_0> | x[1] <= <value_1>
            # and others for each <value_i> with the shape:
            # <value_i> ::= <<first unique value in x[0]>> | <<second unique...
            num_processed_features = 0
            first_feature = params['FITNESS_FUNCTION'].get_first_numerical_feature()
            inputs = params['FITNESS_FUNCTION'].training_in
            if first_feature is not None:
                grammar_content.append('\n<GE_GENERATE:dataset_lessequal_conditions> ::= ')
                # Go over the features of the dataset.
                # This code assumes params['FITNESS_FUNCTION'] is a supervised_learning.supervised_learning object
                header = '(x[\"\'' if isinstance(inputs, pd.DataFrame) else '(x[:,'
                tail = '\'\"]' if isinstance(inputs, pd.DataFrame) else ']'
                grammar_content.append(header + inputs.columns[first_feature] + tail + ' <= <values_feature_' + str(first_feature) + '>)')
                for i in range(first_feature + 1, inputs.shape[1]):
                    if not params['FITNESS_FUNCTION'].is_ithfeature_categorical(i):
                        grammar_content.append(' | ' + header + inputs.columns[i] + tail + ' <= <values_feature_' + str(i) + '>)')

                grammar_content.append('\n')

                for i in range(first_feature, inputs.shape[1]):
                    if not params['FITNESS_FUNCTION'].is_ithfeature_categorical(i):
                        self._generate_values_feature(i, grammar_content)
                        num_processed_features += 1

            self.ge_generate_tags['less_equal_processed_features'] = num_processed_features

        return self.ge_generate_tags['less_equal_processed_features']

    def check_depths(self):
        """
        Run through a grammar and find out the minimum distance from each
        NT to the nearest T. Useful for initialisation methods where we
        need to know how far away we are from fully expanding a tree
        relative to where we are in the tree and what the depth limit is.

        :return: Nothing.
        """

        # Initialise graph and counter for checking minimum steps to Ts for
        # each NT.
        counter, graph = 1, []

        for rule in sorted(self.rules.keys()):
            # Iterate over all NTs.
            choices = self.rules[rule]['choices']

            # Set branching factor for each NT.
            self.non_terminals[rule]['b_factor'] = self.rules[rule][
                'no_choices']

            for choice in choices:
                # Add a new edge to our graph list.
                graph.append([rule, choice['choice']])

        while graph:
            removeset = set()
            for edge in graph:
                # Find edges which either connect to terminals or nodes
                # which are fully expanded.
                if all([sy["type"] == "T" or
                        self.non_terminals[sy["symbol"]]['expanded']
                        for sy in edge[1]]):
                    removeset.add(edge[0])

            for s in removeset:
                # These NTs are now expanded and have their correct minimum
                # path set.
                self.non_terminals[s]['expanded'] = True
                self.non_terminals[s]['min_steps'] = counter

            # Create new graph list and increment counter.
            graph = [e for e in graph if e[0] not in removeset]
            counter += 1

    def check_recursion(self, cur_symbol, seen):
        """
        Traverses the grammar recursively and sets the properties of each rule.

        :param cur_symbol: symbol to check.
        :param seen: Contains already checked symbols in the current traversal.
        :return: Boolean stating whether or not cur_symbol is recursive.
        """

        if cur_symbol not in self.non_terminals.keys():
            # Current symbol is a T.
            return False

        if cur_symbol in seen:
            # Current symbol has already been seen, is recursive.
            return True

        # Append current symbol to seen list.
        seen.append(cur_symbol)

        # Get choices of current symbol.
        choices = self.rules[cur_symbol]['choices']
        nt = self.non_terminals[cur_symbol]

        recursive = False
        for choice in choices:
            for sym in choice['choice']:
                # Recurse over choices.
                recursive_symbol = self.check_recursion(sym["symbol"], seen)
                recursive = recursive or recursive_symbol

        # Set recursive properties.
        nt['recursive'] = recursive
        seen.remove(cur_symbol)

        return nt['recursive']

    def set_arity(self):
        """
        Set the minimum path of the grammar, i.e. the smallest legal
        solution that can be generated.

        Set the maximum arity of the grammar, i.e. the longest path to a
        terminal from any non-terminal.

        :return: Nothing
        """

        # Set the minimum path of the grammar as the minimum steps to a
        # terminal from the start rule.
        self.min_path = self.non_terminals[self.start_rule["symbol"]][
            'min_steps']

        # Set the maximum arity of the grammar as the longest path to
        # a T from any NT.
        self.max_arity = max(self.non_terminals[NT]['min_steps']
                             for NT in self.non_terminals)

        # Add the minimum terminal path to each production rule.
        for rule in self.rules:
            for choice in self.rules[rule]['choices']:
                NT_kids = [i for i in choice['choice'] if i["type"] == "NT"]
                if NT_kids:
                    choice['NT_kids'] = True
                    for sym in NT_kids:
                        sym['min_steps'] = self.non_terminals[sym["symbol"]][
                            'min_steps']

        # Add boolean flag indicating recursion to each production rule.
        for rule in self.rules:
            for prod in self.rules[rule]['choices']:
                for sym in [i for i in prod['choice'] if i["type"] == "NT"]:
                    sym['recursive'] = self.non_terminals[sym["symbol"]][
                        'recursive']
                    if sym['recursive']:
                        prod['recursive'] = True

    def set_grammar_properties(self):
        """
        Goes through all non-terminals and finds the production choices with
        the minimum steps to terminals and with recursive steps.

        :return: Nothing
        """

        for nt in self.non_terminals:
            # Loop over all non terminals.
            # Find the production choices for the current NT.
            choices = self.rules[nt]['choices']

            for choice in choices:
                # Set the maximum path to a terminal for each produciton choice
                choice['max_path'] = max([item["min_steps"] for item in
                                          choice['choice']])

            # Find shortest path to a terminal for all production choices for
            # the current NT. The shortest path will be the minimum of the
            # maximum paths to a T for each choice over all choices.
            min_path = min([choice['max_path'] for choice in choices])

            # Set the minimum path in the self.non_terminals dict.
            self.non_terminals[nt]['min_path'] = [choice for choice in
                                                  choices if choice[
                                                      'max_path'] == min_path]

            # Find recursive production choices for current NT. If any
            # constituent part of a production choice is recursive,
            # it is added to the recursive list.
            self.non_terminals[nt]['recursive'] = [choice for choice in
                                                   choices if choice[
                                                       'recursive']]

    def check_permutations(self):
        """
        Calculates how many possible derivation tree combinations can be
        created from the given grammar at a specified depth. Only returns
        possible combinations at the specific given depth (if there are no
        possible permutations for a given depth, will return 0).

        :param ramps:
        :return: Nothing.
        """

        # Set the number of depths permutations are calculated for
        # (starting from the minimum path of the grammar)
        ramps = params['PERMUTATION_RAMPS']

        perms_list = []
        if self.max_arity > self.min_path:
            for i in range(max((self.max_arity + 1 - self.min_path), ramps)):
                x = self.check_all_permutations(i + self.min_path)
                perms_list.append(x)
                if i > 0:
                    perms_list[i] -= sum(perms_list[:i])
                    self.permutations[i + self.min_path] -= sum(perms_list[:i])
        else:
            for i in range(ramps):
                x = self.check_all_permutations(i + self.min_path)
                perms_list.append(x)
                if i > 0:
                    perms_list[i] -= sum(perms_list[:i])
                    self.permutations[i + self.min_path] -= sum(perms_list[:i])

    def check_all_permutations(self, depth):
        """
        Calculates how many possible derivation tree combinations can be
        created from the given grammar at a specified depth. Returns all
        possible combinations at the specific given depth including those
        depths below the given depth.

        :param depth: A depth for which to calculate the number of
        permutations of solution that can be generated by the grammar.
        :return: The permutations possible at the given depth.
        """

        if depth < self.min_path:
            # There is a bug somewhere that is looking for a tree smaller than
            # any we can create
            s = "representation.grammar.Grammar.check_all_permutations\n" \
                "Error: cannot check permutations for tree smaller than the " \
                "minimum size."
            raise Exception(s)

        if depth in self.permutations.keys():
            # We have already calculated the permutations at the requested
            # depth.
            return self.permutations[depth]

        else:
            # Calculate permutations at the requested depth.
            # Initialise empty data arrays.
            pos, depth_per_symbol_trees, productions = 0, {}, []

            for NT in self.non_terminals:
                # Iterate over all non-terminals to fill out list of
                # productions which contain non-terminal choices.
                a = self.non_terminals[NT]

                for rule in self.rules[a['id']]['choices']:
                    if rule['NT_kids']:
                        productions.append(rule)

            # Get list of all production choices from the start symbol.
            start_symbols = self.rules[self.start_rule["symbol"]]['choices']

            for choice in productions:
                # Generate a list of the symbols of each production choice
                key = str([sym['symbol'] for sym in choice['choice']])

                # Initialise permutations dictionary with the list
                depth_per_symbol_trees[key] = {}

            for i in range(2, depth + 1):
                # Find all the possible permutations from depth of min_path up
                # to a specified depth

                for choice in productions:
                    # Iterate over all production choices
                    sym_pos = 1

                    for j in choice['choice']:
                        # Iterate over all symbols in a production choice.
                        symbol_arity_pos = 0

                        if j["type"] == "NT":
                            # We are only interested in non-terminal symbols
                            for child in self.rules[j["symbol"]]['choices']:
                                # Iterate over all production choices for
                                # each NT symbol in the original choice.

                                if len(child['choice']) == 1 and \
                                        child['choice'][0]["type"] == "T":
                                    # If the child choice leads directly to
                                    # a single terminal, increment the
                                    # permutation count.
                                    symbol_arity_pos += 1

                                else:
                                    # The child choice does not lead
                                    # directly to a single terminal.
                                    # Generate a key for the permutations
                                    # dictionary and increment the
                                    # permutations count there.
                                    key = [sym['symbol'] for sym in
                                           child['choice']]
                                    if (i - 1) in depth_per_symbol_trees[
                                        str(key)].keys():
                                        symbol_arity_pos += \
                                            depth_per_symbol_trees[str(key)][
                                                i - 1]

                            # Multiply original count by new count.
                            sym_pos *= symbol_arity_pos

                    # Generate new key for the current production choice and
                    # set the new value in the permutations dictionary.
                    key = [sym['symbol'] for sym in choice['choice']]
                    depth_per_symbol_trees[str(key)][i] = sym_pos

            # Calculate permutations for the start symbol.
            for sy in start_symbols:
                key = [sym['symbol'] for sym in sy['choice']]
                if str(key) in depth_per_symbol_trees:
                    pos += depth_per_symbol_trees[str(key)][depth] if depth in \
                                                                      depth_per_symbol_trees[
                                                                          str(
                                                                              key)] else 0
                else:
                    pos += 1

            # Set the overall permutations dictionary for the current depth.
            self.permutations[depth] = pos

            return pos

    def get_min_ramp_depth(self):
        """
        Find the minimum depth at which ramping can start where we can have
        unique solutions (no duplicates).

        :param self: An instance of the representation.grammar.grammar class.
        :return: The minimum depth at which unique solutions can be generated
        """

        max_tree_depth = params['MAX_INIT_TREE_DEPTH']
        size = params['POPULATION_SIZE']

        # Specify the range of ramping depths
        depths = range(self.min_path, max_tree_depth + 1)

        if size % 2:
            # Population size is odd
            size += 1

        if size / 2 < len(depths):
            # The population size is too small to fully cover all ramping
            # depths. Only ramp to the number of depths we can reach.
            depths = depths[:int(size / 2)]

        # Find the minimum number of unique solutions required to generate
        # sufficient individuals at each depth.
        unique_start = int(floor(size / len(depths)))
        ramp = None

        for i in sorted(self.permutations.keys()):
            # Examine the number of permutations and combinations of unique
            # solutions capable of being generated by a grammar across each
            # depth i.
            if self.permutations[i] > unique_start:
                # If the number of permutations possible at a given depth i is
                # greater than the required number of unique solutions,
                # set the minimum ramp depth and break out of the loop.
                ramp = i
                break
        self.min_ramp = ramp

    def find_concatenation_NTs(self):
        """
        Scour the grammar class to find non-terminals which can be used to
        combine/reduce_trees derivation trees. Build up a list of such
        non-terminals. A concatenation non-terminal is one in which at least
        one production choice contains multiple non-terminals. For example:

            <e> ::= (<e><o><e>)|<v>

        is a concatenation NT, since the production choice (<e><o><e>) can
        reduce_trees multiple NTs together. Note that this choice also includes
        a combination of terminals and non-terminals.

        :return: Nothing.
        """

        # Iterate over all non-terminals/production rules.
        for rule in sorted(self.rules.keys()):

            # Find rules which have production choices leading to NTs.
            concat = [choice for choice in self.rules[rule]['choices'] if
                      choice['NT_kids']]

            if concat:
                # We can reduce_trees NTs.
                for choice in concat:

                    symbols = [[sym['symbol'], sym['type']] for sym in
                               choice['choice']]

                    NTs = [sym['symbol'] for sym in choice['choice'] if
                           sym['type'] == "NT"]

                    for NT in NTs:
                        # We add to our self.concat_NTs dictionary. The key is
                        # the root node we want to reduce_trees with another
                        # node. This way when we have a node and wish to see
                        # if we can reduce_trees it with anything else, we
                        # simply look up this dictionary.
                        conc = [choice['choice'], rule, symbols]

                        if NT not in self.concat_NTs:
                            self.concat_NTs[NT] = [conc]
                        else:
                            if conc not in self.concat_NTs[NT]:
                                self.concat_NTs[NT].append(conc)

    def __str__(self):
        return "%s %s %s %s" % (self.terminals, self.non_terminals,
                                self.rules, self.start_rule)
