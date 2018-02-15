from __future__ import division

import numpy as np
from collections import Counter
import time
import random
import math


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.

        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.

        Args:
            feature (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.

    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.

    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as numpy array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if class_index == -1:
        classes = map(int, out[:, class_index])
        features = out[:, :class_index]
        return features, classes

    elif class_index == 0:
        classes = map(int, out[:, class_index])
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the provided data.

    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = DecisionNode(None, None, lambda x: x[0] == 1)
    decision_tree_root.left = DecisionNode(None, None, None, 1)

    decision_tree_root.right = DecisionNode(None, None, lambda x: x[2] == 1)
    decision_tree_root.right.left = DecisionNode(None, None, lambda x: x[3] == 1)
    decision_tree_root.right.right = DecisionNode(None, None, lambda x: x[3] == 1)

    decision_tree_root.right.left.left = DecisionNode(None, None, None, 1)
    decision_tree_root.right.left.right = DecisionNode(None, None, None, 0)

    decision_tree_root.right.right.left = DecisionNode(None, None, None, 0)
    decision_tree_root.right.right.right = DecisionNode(None, None, None, 1)

    return decision_tree_root

    # TODO: finish this.
    raise NotImplemented()


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        A two dimensional array representing the confusion matrix.
    """

    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0

    for i in range(0, len(true_labels)):
        if classifier_output[i] == 1:
            if true_labels[i] == classifier_output[i]:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if true_labels[i] == classifier_output[i]:
                true_negative += 1
            else:
                false_negative += 1

    matrix = [[true_positive, false_negative], [false_positive, true_negative]]

    return matrix

    # TODO: finish this.
    raise NotImplemented()


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.

    Precision is measured as:
        true_positive/ (true_positive + false_positive)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The precision of the classifier output.
    """

    true_positive = 0

    for i in range(0, len(true_labels)):
        if classifier_output[i] == 1:
            if true_labels[i] == classifier_output[i]:
                true_positive += 1

    output = true_positive / sum(classifier_output)

    return output

    # TODO: finish this.
    raise NotImplemented()


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.

    Recall is measured as:
        true_positive/ (true_positive + false_negative)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The recall of the classifier output.
    """

    true_positive = 0
    false_negative = 0

    for i in range(0, len(true_labels)):
        if classifier_output[i] == 1:
            if true_labels[i] == classifier_output[i]:
                true_positive += 1
        else:
            if not true_labels[i] == classifier_output[i]:
                false_negative += 1

    output = true_positive / (true_positive + false_negative)

    return output

    # TODO: finish this.
    raise NotImplemented()


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.

    Accuracy is measured as:
        correct_classifications / total_number_examples

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The accuracy of the classifier output.
    """

    true_positive = 0
    true_negative = 0

    for i in range(0, len(true_labels)):
        if classifier_output[i] == 1:
            if true_labels[i] == classifier_output[i]:
                true_positive += 1
        else:
            if true_labels[i] == classifier_output[i]:
                true_negative += 1

    output = (true_positive + true_negative) / len(true_labels)

    return output

    # TODO: finish this.
    raise NotImplemented()


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.

    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.

    Returns:
        Floating point number representing the gini impurity.
    """

    p = sum(class_vector)
    n = len(class_vector) - p
    output = 1 - ((p / (p + n)) ** 2 + (n / (p + n)) ** 2)

    return output

    raise NotImplemented()


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """

    p = sum(previous_classes)
    n = len(previous_classes) - p
    gini_prev = 1 - ((p / (p + n)) ** 2 + (n / (p + n)) ** 2)

    output = gini_prev

    for split in current_classes:
        pp = sum(split)
        nn = len(split) - pp
        gini = 1 - ((pp / (pp + nn)) ** 2 + (nn / (pp + nn)) ** 2)
        output -= ((pp+nn)/(p+n))*gini

    return output

    raise NotImplemented()


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=50):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """

        try:
            p = sum(classes)
            n = len(classes) - p

            if n == 0:
                decision_tree_root = DecisionNode(None, None, None, 1)

            elif p == 0:
                decision_tree_root = DecisionNode(None, None, None, 0)

            elif depth > 0:
                best_feature = -1
                max_gini_gain = -1
                alpha = 0

                feature_class0 = []
                feature_class1 = []
                feature_class2 = []
                feature_class3 = []

                i = 0
                for x in features:
                    feature_class0.append([x[0], classes[i]])
                    feature_class1.append([x[1], classes[i]])
                    feature_class2.append([x[2], classes[i]])
                    feature_class3.append([x[3], classes[i]])
                    i += 1

                sorted_feature_class0 = sorted(feature_class0)
                sorted_feature_class1 = sorted(feature_class1)
                sorted_feature_class2 = sorted(feature_class2)
                sorted_feature_class3 = sorted(feature_class3)

                sortedclasses0 = [x[1] for x in sorted_feature_class0]
                sortedclasses1 = [x[1] for x in sorted_feature_class1]
                sortedclasses2 = [x[1] for x in sorted_feature_class2]
                sortedclasses3 = [x[1] for x in sorted_feature_class3]

                sortedfeature0 = [x[0] for x in sorted_feature_class0]
                sortedfeature1 = [x[0] for x in sorted_feature_class1]
                sortedfeature2 = [x[0] for x in sorted_feature_class2]
                sortedfeature3 = [x[0] for x in sorted_feature_class3]

                for i in range(1, len(classes)):

                    current_classes = [sortedclasses0[:i], sortedclasses0[i:]]
                    gain = gini_gain(sortedclasses0, current_classes)
                    if gain >= max_gini_gain:
                        max_gini_gain = gain
                        best_feature = 0
                        alpha = sortedfeature0[i]

                    current_classes = [sortedclasses1[:i], sortedclasses1[i:]]
                    gain = gini_gain(sortedclasses1, current_classes)
                    if gain >= max_gini_gain:
                        max_gini_gain = gain
                        best_feature = 1
                        alpha = sortedfeature1[i]

                    current_classes = [sortedclasses2[:i], sortedclasses2[i:]]
                    gain = gini_gain(sortedclasses2, current_classes)
                    if gain >= max_gini_gain:
                        max_gini_gain = gain
                        best_feature = 2
                        alpha = sortedfeature2[i]

                    current_classes = [sortedclasses3[:i], sortedclasses3[i:]]
                    gain = gini_gain(sortedclasses3, current_classes)
                    if gain >= max_gini_gain:
                        max_gini_gain = gain
                        best_feature = 3
                        alpha = sortedfeature3[i]

                decision_tree_root = DecisionNode(None, None, lambda y: y[best_feature] <= alpha)

                left_features = []
                right_features = []
                left_classes = []
                right_classes = []
                for j in range(len(features)):
                    if features[j][best_feature] <= alpha:
                        left_features.append(features[j])
                        left_classes.append(classes[j])
                    else:
                        right_features.append(features[j])
                        right_classes.append(classes[j])

                decision_tree_root.left = self.__build_tree__(left_features, left_classes, depth - 1)
                decision_tree_root.right = self.__build_tree__(right_features, right_classes, depth - 1)

            else:
                if p > n:
                    decision_tree_root = DecisionNode(None, None, None, 1)
                else:
                    decision_tree_root = DecisionNode(None, None, None, 0)

            return decision_tree_root

        except ValueError:
            decision_tree_root = DecisionNode(None, None, None, 0)
            return decision_tree_root

        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.

        Args:
            features (list(list(int)): List of features.

        Return:
            A list of class labels.
        """

        class_labels = []

        for x in features:
            result = self.root.decide(x)
            class_labels.append(result)

        return class_labels

        # TODO: finish this.
        raise NotImplemented()


def generate_k_folds(dataset, k):
    """Split dataset into folds.

    Randomly split data into k equal subsets.

    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).

    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.

    Returns:
        List of folds.
    """

    output = []

    dataset = list(dataset)
    random.shuffle(dataset)
    n = len(dataset[0])//k

    for i in range(k):
        training_examples = []
        training_classes = []
        testing_examples = []
        testing_classes = []
        for j in range(len(dataset[0])):
            if j in range(i*n, (i+1)*n):
                testing_examples.append(dataset[0][j])
                testing_classes.append(dataset[1][j])
            else:
                training_examples.append(dataset[0][j])
                training_classes.append(dataset[1][j])
        training_set = (training_examples, training_classes)
        testing_set = (testing_examples, testing_classes)
        fold = (training_set, testing_set)
        output.append(fold)

    return output

    # TODO: finish this.
    raise NotImplemented()


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.

         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.root = [None, None, None, None, None]

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.

            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        for j in range(self.num_trees):
            feature_class = list([])

            i = 0
            for x in features:
                c = classes[i]
                y = list(x)
                y.append(c)
                feature_class.append(y)
                i += 1

            for l in range(20):
                random.shuffle(feature_class)

            num_examples = int(len(features)*self.example_subsample_rate)
            # num_features = int(len(features[0])*self.attr_subsample_rate)
            # for l in range(3):
            #     feature_index = random.sample(range(len(features[0])), num_features)

            if j == 0:
                feature_index = [0, 1]
            if j == 1:
                feature_index = [1, 2]
            if j == 2:
                feature_index = [2, 3]
            if j == 3:
                feature_index = [0, 2]
            if j == 4:
                feature_index = [1, 3]

            features1 = []
            classes1 = []
            for x in feature_class[0:num_examples]:
                feature = []
                for i in feature_index:
                    feature.append(x[i])
                features1.append(feature)
                classes1.append(x[4])

            self.root[j] = self.__build_tree__(features1, classes1, self.depth_limit)

        # # TODO: finish this.
        # raise NotImplemented()

    def __build_tree__(self, features, classes, depth=5):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """

        p = sum(classes)
        n = len(classes) - p

        if n == 0:
            decision_tree_root = DecisionNode(None, None, None, 1)

        elif p == 0:
            decision_tree_root = DecisionNode(None, None, None, 0)

        elif depth > 0:
            best_feature = -1
            max_gini_gain = -1
            alpha = 0

            feature_class0 = []
            feature_class1 = []

            i = 0
            for x in features:
                feature_class0.append([x[0], classes[i]])
                feature_class1.append([x[1], classes[i]])
                i += 1

            sorted_feature_class0 = sorted(feature_class0)
            sorted_feature_class1 = sorted(feature_class1)

            sortedclasses0 = [x[1] for x in sorted_feature_class0]
            sortedclasses1 = [x[1] for x in sorted_feature_class1]

            sortedfeature0 = [x[0] for x in sorted_feature_class0]
            sortedfeature1 = [x[0] for x in sorted_feature_class1]

            for i in range(1, len(classes)):

                current_classes = [sortedclasses0[:i], sortedclasses0[i:]]
                gain = gini_gain(sortedclasses0, current_classes)
                if gain >= max_gini_gain:
                    max_gini_gain = gain
                    best_feature = 0
                    alpha = sortedfeature0[i]

                current_classes = [sortedclasses1[:i], sortedclasses1[i:]]
                gain = gini_gain(sortedclasses1, current_classes)
                if gain >= max_gini_gain:
                    max_gini_gain = gain
                    best_feature = 1
                    alpha = sortedfeature1[i]

            decision_tree_root = DecisionNode(None, None, lambda y: y[best_feature] <= alpha)

            left_features = []
            right_features = []
            left_classes = []
            right_classes = []
            for j in range(len(features)):
                if features[j][best_feature] <= alpha:
                    left_features.append(features[j])
                    left_classes.append(classes[j])
                else:
                    right_features.append(features[j])
                    right_classes.append(classes[j])

            decision_tree_root.left = self.__build_tree__(left_features, left_classes, depth - 1)
            decision_tree_root.right = self.__build_tree__(right_features, right_classes, depth - 1)

        else:
            if p > n:
                decision_tree_root = DecisionNode(None, None, None, 1)
            else:
                decision_tree_root = DecisionNode(None, None, None, 0)

        return decision_tree_root

    def classify(self, features):
        """Classify a list of features based on the trained random forest.

        Args:
            features (list(list(int)): List of features.
        """

        positive_votes = np.zeros(len(features))
        for i in range(self.num_trees):
            # class_labels = []
            j = 0
            for x in features:
                result = self.root[i].decide(x)
                # class_labels.append(result)
                if result == 1:
                    positive_votes[j] += 1
                j += 1

        output = []
        i = 0
        for votes in positive_votes:
            if votes > self.num_trees/2:
                output.append(1)
            else:
                output.append(0)
            i += 1

        return output

        # # TODO: finish this.
        # raise NotImplemented()


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        """Create challenge classifier.

        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # self.trees = []
        # self.num_trees = 9
        # self.depth_limit = 5
        # self.example_subsample_rate = 0.5
        # self.attr_subsample_rate = 0.2
        # self.root = [None, None, None, None, None, None, None, None, None]

        # # TODO: finish this.
        # raise NotImplemented()

    def fit(self, features, classes):
        """Build the underlying tree(s).

            Fit your model to the provided features.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # for j in range(self.num_trees):
        #     feature_class = list([])
        #
        #     i = 0
        #     for x in features:
        #         c = classes[i]
        #         y = list(x)
        #         y.append(c)
        #         feature_class.append(y)
        #         i += 1
        #
        #     # for l in range(20):
        #     #     random.shuffle(feature_class)
        #
        #     num_examples = int(len(features)*self.example_subsample_rate)
        #     # num_features = int(len(features[0])*self.attr_subsample_rate)
        #     # for l in range(3):
        #     #     feature_index = random.sample(range(len(features[0])), num_features)
        #
        #     if j == 0:
        #         feature_index = range(6)
        #     if j == 1:
        #         feature_index = range(6, 12)
        #     if j == 2:
        #         feature_index = range(12, 18)
        #     if j == 3:
        #         feature_index = range(18, 24)
        #     if j == 4:
        #         feature_index = range(24, 30)
        #     if j == 5:
        #         feature_index = [5, 6, 7, 14, 24, 25]
        #     if j == 6:
        #         feature_index = [8, 17, 26, 13, 7, 12]
        #     if j == 7:
        #         feature_index = [9, 3, 11, 7, 18, 29]
        #     if j == 8:
        #         feature_index = [1, 15, 18, 5, 16, 26]
        #
        #     features1 = []
        #     classes1 = []
        #     if j % 2 == 0:
        #         for x in feature_class[0:num_examples]:
        #             feature = []
        #             for i in feature_index:
        #                 feature.append(x[i])
        #             features1.append(feature)
        #             classes1.append(x[30])
        #     else:
        #         for x in feature_class[num_examples:]:
        #             feature = []
        #             for i in feature_index:
        #                 feature.append(x[i])
        #             features1.append(feature)
        #             classes1.append(x[30])
        #
        #     self.root[j] = self.__build_tree__(features1, classes1, self.depth_limit)

        # # TODO: finish this.
        # raise NotImplemented()

    def __build_tree__(self, features, classes, depth=30):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """

        # p = sum(classes)
        # n = len(classes) - p
        #
        # if n == 0:
        #     decision_tree_root = DecisionNode(None, None, None, 1)
        #
        # elif p == 0:
        #     decision_tree_root = DecisionNode(None, None, None, 0)
        #
        # elif depth > 0:
        #     best_feature = -1
        #     max_gini_gain = -1
        #     alpha = 0
        #
        #     feature_class0 = []
        #     feature_class1 = []
        #     feature_class2 = []
        #     feature_class3 = []
        #     feature_class4 = []
        #     feature_class5 = []
        #     # feature_class6 = []
        #     # feature_class7 = []
        #     # feature_class8 = []
        #
        #
        #     i = 0
        #     for x in features:
        #         feature_class0.append([x[0], classes[i]])
        #         feature_class1.append([x[1], classes[i]])
        #         feature_class2.append([x[2], classes[i]])
        #         feature_class3.append([x[3], classes[i]])
        #         feature_class4.append([x[4], classes[i]])
        #         feature_class5.append([x[5], classes[i]])
        #         # feature_class6.append([x[6], classes[i]])
        #         # feature_class7.append([x[7], classes[i]])
        #         # feature_class8.append([x[8], classes[i]])
        #         i += 1
        #
        #     sorted_feature_class0 = sorted(feature_class0)
        #     sorted_feature_class1 = sorted(feature_class1)
        #     sorted_feature_class2 = sorted(feature_class2)
        #     sorted_feature_class3 = sorted(feature_class3)
        #     sorted_feature_class4 = sorted(feature_class4)
        #     sorted_feature_class5 = sorted(feature_class5)
        #     # sorted_feature_class6 = sorted(feature_class6)
        #     # sorted_feature_class7 = sorted(feature_class7)
        #     # sorted_feature_class8 = sorted(feature_class8)
        #
        #     sortedclasses0 = [x[1] for x in sorted_feature_class0]
        #     sortedclasses1 = [x[1] for x in sorted_feature_class1]
        #     sortedclasses2 = [x[1] for x in sorted_feature_class2]
        #     sortedclasses3 = [x[1] for x in sorted_feature_class3]
        #     sortedclasses4 = [x[1] for x in sorted_feature_class4]
        #     sortedclasses5 = [x[1] for x in sorted_feature_class5]
        #     # sortedclasses6 = [x[1] for x in sorted_feature_class6]
        #     # sortedclasses7 = [x[1] for x in sorted_feature_class7]
        #     # sortedclasses8 = [x[1] for x in sorted_feature_class8]
        #
        #     sortedfeature0 = [x[0] for x in sorted_feature_class0]
        #     sortedfeature1 = [x[0] for x in sorted_feature_class1]
        #     sortedfeature2 = [x[0] for x in sorted_feature_class2]
        #     sortedfeature3 = [x[0] for x in sorted_feature_class3]
        #     sortedfeature4 = [x[0] for x in sorted_feature_class4]
        #     sortedfeature5 = [x[0] for x in sorted_feature_class5]
        #     # sortedfeature6 = [x[0] for x in sorted_feature_class6]
        #     # sortedfeature7 = [x[0] for x in sorted_feature_class7]
        #     # sortedfeature8 = [x[0] for x in sorted_feature_class8]
        #
        #     for i in range(1, len(classes)):
        #
        #         current_classes = [sortedclasses0[:i], sortedclasses0[i:]]
        #         gain = gini_gain(sortedclasses0, current_classes)
        #         if gain >= max_gini_gain:
        #             max_gini_gain = gain
        #             best_feature = 0
        #             alpha = sortedfeature0[i]
        #
        #         current_classes = [sortedclasses1[:i], sortedclasses1[i:]]
        #         gain = gini_gain(sortedclasses1, current_classes)
        #         if gain >= max_gini_gain:
        #             max_gini_gain = gain
        #             best_feature = 1
        #             alpha = sortedfeature1[i]
        #
        #         current_classes = [sortedclasses2[:i], sortedclasses2[i:]]
        #         gain = gini_gain(sortedclasses2, current_classes)
        #         if gain >= max_gini_gain:
        #             max_gini_gain = gain
        #             best_feature = 2
        #             alpha = sortedfeature2[i]
        #
        #         current_classes = [sortedclasses3[:i], sortedclasses3[i:]]
        #         gain = gini_gain(sortedclasses3, current_classes)
        #         if gain >= max_gini_gain:
        #             max_gini_gain = gain
        #             best_feature = 3
        #             alpha = sortedfeature3[i]
        #
        #         current_classes = [sortedclasses4[:i], sortedclasses4[i:]]
        #         gain = gini_gain(sortedclasses4, current_classes)
        #         if gain >= max_gini_gain:
        #             max_gini_gain = gain
        #             best_feature = 4
        #             alpha = sortedfeature4[i]
        #
        #         current_classes = [sortedclasses5[:i], sortedclasses5[i:]]
        #         gain = gini_gain(sortedclasses5, current_classes)
        #         if gain >= max_gini_gain:
        #             max_gini_gain = gain
        #             best_feature = 5
        #             alpha = sortedfeature5[i]

                # current_classes = [sortedclasses6[:i], sortedclasses6[i:]]
                # gain = gini_gain(sortedclasses6, current_classes)
                # if gain >= max_gini_gain:
                #     max_gini_gain = gain
                #     best_feature = 6
                #     alpha = sortedfeature6[i]
                #
                # current_classes = [sortedclasses7[:i], sortedclasses7[i:]]
                # gain = gini_gain(sortedclasses7, current_classes)
                # if gain >= max_gini_gain:
                #     max_gini_gain = gain
                #     best_feature = 7
                #     alpha = sortedfeature7[i]
                #
                # current_classes = [sortedclasses8[:i], sortedclasses8[i:]]
                # gain = gini_gain(sortedclasses8, current_classes)
                # if gain >= max_gini_gain:
                #     max_gini_gain = gain
                #     best_feature = 8
                #     alpha = sortedfeature8[i]

        #     decision_tree_root = DecisionNode(None, None, lambda y: y[best_feature] <= alpha)
        #
        #     left_features = []
        #     right_features = []
        #     left_classes = []
        #     right_classes = []
        #     for j in range(len(features)):
        #         if features[j][best_feature] <= alpha:
        #             left_features.append(features[j])
        #             left_classes.append(classes[j])
        #         else:
        #             right_features.append(features[j])
        #             right_classes.append(classes[j])
        #
        #     decision_tree_root.left = self.__build_tree__(left_features, left_classes, depth - 1)
        #     decision_tree_root.right = self.__build_tree__(right_features, right_classes, depth - 1)
        #
        # else:
        #     if p > n:
        #         decision_tree_root = DecisionNode(None, None, None, 1)
        #     else:
        #         decision_tree_root = DecisionNode(None, None, None, 0)
        #
        # return decision_tree_root

    def classify(self, features):
        """Classify a list of features.

        Classify each feature in features as either 0 or 1.

        Args:
            features (list(list(int)): List of features.

        Returns:
            A list of class labels.
        """

        # positive_votes = []
        # for i in range(len(features)):
        #     positive_votes.append(0)
        #
        # for i in range(self.num_trees):
        #     if i == 0:
        #         feature_index = range(6)
        #     if i == 1:
        #         feature_index = range(6, 12)
        #     if i == 2:
        #         feature_index = range(12, 18)
        #     if i == 3:
        #         feature_index = range(18, 24)
        #     if i == 4:
        #         feature_index = range(24, 30)
        #     if i == 5:
        #         feature_index = [5, 6, 7, 14, 24, 25]
        #     if i == 6:
        #         feature_index = [8, 17, 26, 13, 7, 12]
        #     if i == 7:
        #         feature_index = [9, 3, 11, 7, 18, 29]
        #     if i == 8:
        #         feature_index = [1, 15, 18, 5, 16, 26]
        #
        #     features1 = []
        #     for x in features:
        #         feature = []
        #         for k in feature_index:
        #             feature.append(x[k])
        #         features1.append(feature)
        #
        #     j = 0
        #     for x in features1:
        #         result = self.root[i].decide(x)
        #         if result == 1:
        #             positive_votes[j] += 1
        #         j += 1
        #
        # output = []
        # for votes in positive_votes:
        #     if votes > self.num_trees / 2:
        #         output.append(1)
        #     else:
        #         output.append(0)
        #
        # return output

        # # TODO: finish this.
        # raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Args:
            data: data to be added to array.

        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])

        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Bonnie time to beat: 0.09 seconds.

        Args:
            data: data to be sliced and summed.

        Returns:
            Numpy array of data.
        """

        row = data.shape[0]
        col = data.shape[1]
        n = row*col

        data = data.reshape([n, 1])
        output = data**2 + data

        output = output.reshape([row, col])

        return output

        # TODO: finish this.
        raise NotImplemented()

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Args:
            data: data to be added to array.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Bonnie time to beat: 0.07 seconds

        Args:
            data: data to be sliced and summed.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        temp = data.sum(axis=1)
        temp = temp[0:100]

        max_sum = temp.max()
        max_row = temp.argmax()

        return max_sum, max_row

        # TODO: finish this.
        raise NotImplemented()

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Bonnie time to beat: 15 seconds

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique = {}
        row = data.shape[0]
        col = data.shape[1]
        data = data.reshape([row * col, 1])

        for i in range(row*col):
            n = data[i].item()
            if n > 0:
                if n in unique:
                    unique[n] += 1
                else:
                    unique[n] = 1

        return unique.items()

        # TODO: finish this.
        raise NotImplemented()
        
def return_your_name():
    # return your name
    return "Dhaneshwaran Jotheeswaran"
    # TODO: finish this
    raise NotImplemented()
