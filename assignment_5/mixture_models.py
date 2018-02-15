from __future__ import division
import warnings
import numpy as np
import scipy as sp
from matplotlib import image
from random import randint
import random
from scipy.misc import logsumexp
from helper_functions import image_to_matrix, matrix_to_image, \
                             flatten_image_matrix, unflatten_image_matrix, \
                             image_difference

warnings.simplefilter(action="ignore", category=FutureWarning)


def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """

    row, col, val = image_values.shape
    vec_img = image_values.copy().reshape([row*col, val])

    if initial_means is None:
        initial_means = random.sample(vec_img, k)

    vec_img_k = vec_img.repeat(k, axis=0).reshape([row*col, k, val])

    conv = np.array([False])

    while not conv.all():
        closest_cluster = ((vec_img_k - initial_means)**2).sum(axis=2).argmin(axis=1)
        sum_pixels = np.zeros([1, k, val])
        num_pixels = np.zeros([1, k, 1])

        i = 0
        for c in closest_cluster:
            sum_pixels[0][c] += vec_img[i]
            num_pixels[0][c] += 1
            i += 1

        k_means = sum_pixels/num_pixels
        conv = k_means == initial_means
        initial_means = k_means.copy()

    i = 0
    for c in closest_cluster:
        vec_img[i] = k_means[0][c]
        i += 1


    output = vec_img.reshape([row, col, val])
    return output

    # TODO: finish this function
    raise NotImplementedError()


def default_convergence(prev_likelihood, new_likelihood, conv_ctr,
                        conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if means is None:
            self.means = np.ndarray([1, num_components])
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)
        self.row, self.col = self.image_matrix.shape
        self.vec_img = self.image_matrix.copy().reshape([self.row * self.col, 1])
        self.vec_img_k = self.vec_img.repeat(self.num_components, axis=1)

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """

        return np.log((self.mixing_coefficients*(1/(2*np.pi*self.variances)**0.5) *
                       np.exp((-0.5*(val - self.means)**2)/self.variances)).sum())

        # # TODO: finish this
        # raise NotImplementedError()

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """

        self.means = (np.array(random.sample(self.vec_img, self.num_components))).transpose()
        self.variances[:] = 1
        self.mixing_coefficients[:] = 1.0/self.num_components

        # # TODO: finish this
        # raise NotImplementedError()

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """

        count = 0
        prev_likelihood = 0
        conv = False

        while not conv:
            # E-Step
            gamma_num = self.mixing_coefficients * (1 / (2 * np.pi * self.variances) ** 0.5) * \
                np.exp((-0.5 * (self.vec_img_k - self.means) ** 2)/self.variances)
            gamma_den = gamma_num.sum(axis=1)

            gamma = np.ndarray([self.row*self.col, self.num_components])
            for i in range(self.num_components):
                gamma[:, i] = gamma_num[:, i]/gamma_den

            # M-Step
            n_k = gamma.sum(axis=0)
            self.means = (gamma*self.vec_img).sum(axis=0)/n_k

            vec = self.vec_img_k-self.means
            self.variances = (gamma*vec*vec).sum(axis=0)/n_k

            i = 0
            for s in self.variances:
                if s < 0.0001:
                    self.variances[i] = 0.5
                    self.means[i] = self.vec_img[randint(0, self.col*self.row)]
                i += 1

            self.mixing_coefficients = n_k/(self.row*self.col)

            # Likelihood
            new_likelihood = self.likelihood()
            count, conv = default_convergence(prev_likelihood, new_likelihood, count, 10)
            # print new_likelihood, self.means
            prev_likelihood = new_likelihood


        # # TODO: finish this
        # raise NotImplementedError()

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """

        likelihood = self.mixing_coefficients * (1 / (2 * np.pi * self.variances) ** 0.5) * \
            np.exp((-0.5 * (self.vec_img_k - self.means) ** 2) / self.variances)

        closest_k = likelihood.argmax(axis=1)

        output = np.ndarray([self.row*self.col, 1])

        i = 0
        for c in closest_k:
            output[i] = self.means[c]
            i += 1

        output = output.reshape([self.row, self.col])
        # im2 = plt.imshow(output)
        return output

        # # TODO: finish this
        # raise NotImplementedError()

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """

        return np.log((self.mixing_coefficients * (1 / (2 * np.pi * self.variances) ** 0.5) *
                       np.exp((-0.5 * (self.vec_img_k - self.means) ** 2) /
                       self.variances)).sum(axis=1)).sum()

        # # TODO: finish this
        # raise NotImplementedError()

    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        best_score = float("-inf")
        for i in range(iters):
            self.initialize_training()
            self.train_model()
            likelihood = self.mixing_coefficients * (1 / (2 * np.pi * self.variances) ** 0.5) * \
                     np.exp((-0.5 * (self.vec_img_k - self.means) ** 2) / self.variances)
            score = likelihood.max(axis=1).sum()
            print score
            if score > best_score:
                means = self.means.copy()
                variances = self.variances.copy()
                mixing_coefficients = self.mixing_coefficients.copy()
                best_score = score

        self.means = means
        self.variances = variances
        self.mixing_coefficients = mixing_coefficients

        likelihood = self.mixing_coefficients * (1 / (2 * np.pi * self.variances) ** 0.5) * \
                     np.exp((-0.5 * (self.vec_img_k - self.means) ** 2) / self.variances)

        closest_k = likelihood.argmax(axis=1)

        output = np.ndarray([self.row * self.col, 1])

        i = 0
        for c in closest_k:
            output[i] = self.means[c]
            i += 1

        output = output.reshape([self.row, self.col])
        # im2 = plt.imshow(output)
        return output

        # # finish this
        # raise NotImplementedError()


class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that
        you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient
         initializations too if that works well.]
        """

        self.variances[:] = 1
        self.mixing_coefficients[:] = 1.0/self.num_components

        # best_likelihood = float("-inf")
        #
        # for n in range(3):
        #     initial_means = (np.array(random.sample(self.vec_img, self.num_components))).transpose()
        #
        #     conv = np.array([False])
        #
        #     while not conv.all():
        #         closest_cluster = ((self.vec_img_k - initial_means) ** 2).argmin(axis=1)
        #         sum_pixels = np.zeros([1, self.num_components])
        #         num_pixels = np.zeros([1, self.num_components])
        #
        #         i = 0
        #         for c in closest_cluster:
        #             sum_pixels[0][c] += self.vec_img[i]
        #             num_pixels[0][c] += 1
        #             i += 1
        #
        #         k_means = sum_pixels / num_pixels
        #         conv = k_means == initial_means
        #         initial_means = k_means.copy()
        #
        #     self.means = np.array(k_means)
        #
        #     likelihood = self.likelihood()
        #     if likelihood > best_likelihood:
        #         means = self.means.copy()
        #         best_likelihood = likelihood
        #
        # self.means = means.copy()

        self.means = np.ndarray([1, self.num_components])
        min_val = min(self.vec_img)
        max_val = max(self.vec_img)

        if self.num_components == 1:
            self.means[0] = (min_val+max_val)/2.0

        else:
            for i in range(self.num_components):
                self.means[0][i] = i*(min_val+max_val)/self.num_components

        # # TODO: finish this
        # raise NotImplementedError()


def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """

    increase_convergence_ctr = ((abs(previous_variables) * 0.97) < abs(new_variables)).all() and \
                               (abs(new_variables) < (abs(previous_variables) * 1.03)).all()

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap

    # # TODO: finish this function
    # raise NotImplementedError()


class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):

        count = 0
        previous_variables = np.array([self.means, self.variances, self.mixing_coefficients])
        conv = False

        while not conv:
            # E-Step
            gamma_num = self.mixing_coefficients * (1 / (2 * np.pi * self.variances) ** 0.5) * \
                        np.exp((-0.5 * (self.vec_img_k - self.means) ** 2) / self.variances)
            gamma_den = gamma_num.sum(axis=1)

            gamma = np.ndarray([self.row * self.col, self.num_components])
            for i in range(self.num_components):
                gamma[:, i] = gamma_num[:, i] / gamma_den

            # M-Step
            n_k = gamma.sum(axis=0)
            self.means = (gamma * self.vec_img).sum(axis=0) / n_k

            vec = self.vec_img_k - self.means
            self.variances = (gamma * vec * vec).sum(axis=0) / n_k

            i = 0
            for s in self.variances:
                if s < 0.0001:
                    self.variances[i] = 0.5
                    self.means[i] = self.vec_img[randint(0, self.col * self.row)]
                i += 1

            self.mixing_coefficients = n_k / (self.row * self.col)

            new_variables = np.array([self.means, self.variances, self.mixing_coefficients])
            count, conv = new_convergence_function(previous_variables, new_variables, count, 100)
            previous_variables = new_variables.copy()

        # # TODO: finish this function
        # raise NotImplementedError()


def bayes_info_criterion(gmm):

    output = (np.log(gmm.vec_img.shape[0]) * 3 * gmm.num_components) - (2 * gmm.likelihood())

    return round(output)

    # # TODO: finish this function
    # raise NotImplementedError()


def BIC_likelihood_model_test():
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel

    for testing purposes:
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]
    ]
    """

    comp_means = np.array([
        np.array([0.023529412, 0.1254902]),
        np.array([0.023529412, 0.1254902, 0.20392157]),
        np.array([0.023529412, 0.1254902, 0.20392157, 0.36078432]),
        np.array([0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689]),
        np.array([0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563]),
        np.array([0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706])
    ])
    min_bic = float("inf")
    min_bic_comps = 2
    max_likelihood_comps = 2
    max_likelihood = float("-inf")
    image_file = 'images/party_spock.png'
    grays = False

    image_matrix = image.imread(image_file)
    # in case of transparency values
    if len(image_matrix.shape) == 3 and image_matrix.shape[2] > 3:
        height, width, depth = image_matrix.shape
        new_img = np.zeros([height, width, 3])
        for r in range(height):
            for c in range(width):
                new_img[r, c, :] = image_matrix[r, c, 0:3]
        image_matrix = np.copy(new_img)
    if grays and len(image_matrix.shape) == 3:
        height, width = image_matrix.shape[0:2]
        new_img = np.zeros([height, width])
        for r in range(height):
            for c in range(width):
                new_img[r, c] = image_matrix[r, c, 0]
        image_matrix = new_img

    for n in range(2, 8):
        initial_means = comp_means[n-2]
        gmm = GaussianMixtureModel(image_matrix, n)
        gmm.initialize_training()
        gmm.means = np.copy(initial_means)
        gmm.train_model()
        likelihood = gmm.likelihood()
        bic = bayes_info_criterion(gmm)

        if bic < min_bic:
            # print "bic", bic, min_bic
            min_bic = bic
            min_bic_comps = n
            min_bic_model = gmm

        if likelihood > max_likelihood:
            # print "likelihood", likelihood, max_likelihood
            max_likelihood = likelihood
            max_likelihood_comps = n
            max_likelihood_model = gmm

    # print min_bic_comps, max_likelihood_comps
    return min_bic_model, max_likelihood_model

    # # TODO: finish this method
    # raise NotImplementedError()


def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    # # TODO: fill in bic and likelihood
    # raise NotImplementedError()
    bic = 7
    likelihood = 7
    pairs = {
        'BIC': bic,
        'likelihood': likelihood
    }
    return pairs

def return_your_name():
    # return your name

    return "Dhaneshwaran Jotheeswaran"

    # # TODO: finish this
    # raise NotImplemented()

def bonus(points_array, means_array):
    """
    Return the distance from every point in points_array
    to every point in means_array.

    returns:
    dists = numpy array of float
    """

    # row, col, val = points_array.shape
    # vec_img = points_array.copy().reshape([row * col, val])
    #
    # vec_img_k = vec_img.repeat(k, axis=0).reshape([row * col, k, val])

    # TODO: fill in the bonus function
    # REMOVE THE LINE BELOW IF ATTEMPTING BONUS
    raise NotImplementedError()
    return dists
