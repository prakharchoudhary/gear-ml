import numpy as np


class DecisionTreeClassifier(object):

    def __init__(self, train, max_depth=None, min_size=2):
        self.train = train
        self.max_depth = max_depth

    def _gini_index(self, groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score of each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def _cross_entropy(self, attr):
        pass

    def _test_split(self, index, value, data):
        left, right = list(), list()
        for row in data:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def _get_split(self, data):
        class_values = list(set(row[-1] for row in data))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(data[0]) - 1):
            for row in dataset:
                groups = self._test_split(index, row[index], data)
                gini = self._gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[
                        index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def _to_terminal(group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def _split(node, max_depth=self.max_depth,
               min_size=self.min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node[
                'right'] = self._to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self._to_terminal(left),\
                self._to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self._to_terminal(left)
        else:
            node['left'] = self._get_split(left)
            self._split(node['left'], depth=depth + 1)
        # process left child
        if len(right) <= min_size:
            node['right'] = self._to_terminal(right)
        else:
            node['right'] = self._get_split(right)
            self._split(node['right'], depth=depth + 1)

    # Build a decision tree
    def build_tree(self, train, max_depth=self.max_depth,
                   min_size=self.min_size):
        root = self._get_split(self.train)
        self._split(root, depth=1)
        return root

    # Make a prediction with a decision tree
    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], dict)
            else:
                return node['right']
