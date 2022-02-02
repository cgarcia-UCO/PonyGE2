import re


def divide_ant_cons(string):
    """
    This function divides a string by the first ',' not in any set of parenthesis.
    (I know that) This can be done with regular expressions. I do not know if this would be more efficient

    :param string: any string, although it is supposed to match the expression .*,.*
    :return: a tuple with the first part of the string and the second one
    """
    i = 0
    num_open_parenthesis = 0

    while i < len(string) and (string[i] != ',' or num_open_parenthesis != 0):
        if string[i] == '(':
            num_open_parenthesis += 1
        elif string[i] == ')':
            num_open_parenthesis -= 1

        i += 1

    return string[:i], string[i + 1:]


def extract_consecuent(string):
    """
    This function extract a list with the consecuents of an association rule.

    :param string: unprocessed string with the consecuents.
    :return: list with the consecuents.

        Example:
            consecuent = " 'Si', 'No')"
            returned value = ['Si', 'No']
    """
    return re.findall("'([^']*)'", string)


def nested_conds_2_rules_list(string):
    """
    This function processes a string with nested constructions of the form: 'np.where(....,....,...)',
    and returns a list with the conjunction of the conditions that precedes any leaf decision

    :param string: any string matching the expression 'np.where(.+,.+,.+)', for instance:
                    np.where((x.iloc[:,3] < 3) & (x.iloc[:,2] != 'Yes'), 'Yes', np.where(x.iloc[:,1] > 0,'No','Yes')
    :return: a list with the conjunction of the conditions that precedes any leaf decision. For instance,
            for the previous example:
            [
            "(x.iloc[:,3] < 3) & (x.iloc[:,2] != 'Yes')",
            "~((x.iloc[:,3] < 3) & (x.iloc[:,2] != 'Yes')) & x.iloc[:,1] > 0",
            "~((x.iloc[:,3] < 3) & (x.iloc[:,2] != 'Yes')) & ~(x.iloc[:,1] > 0)"
            ]
    """
    substrings = string.split('np.where(')
    result = []
    current_antecedent = []
    num_uses = []

    for i in substrings:
        antecedent, consecuent = divide_ant_cons(i)
        current_antecedent.append(antecedent)
        num_uses.append(0)

        if consecuent.strip() != '':
            for j in consecuent.split(','):
                if len(j.strip()) > 0:
                    # + " => " + j)
                    result.append(" & ".join(current_antecedent)[3:])

                    if num_uses[-1] == 0:
                        last_condition = current_antecedent.pop()
                        current_antecedent.append('~(' + last_condition + ')')
                        num_uses[-1] = 1
                    else:

                        while num_uses[-1] > 0:
                            current_antecedent.pop()
                            num_uses.pop()

                        last_condition = current_antecedent.pop()
                        current_antecedent.append('~(' + last_condition + ')')
                        num_uses[-1] = 1

    return result, extract_consecuent(consecuent)
