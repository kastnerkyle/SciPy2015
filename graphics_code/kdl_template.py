# Author: Kyle Kastner
# License: BSD 3-clause
# Ideas from Junyoung Chung and Kyunghyun Cho
# The latest version of this template will always live in:
# https://github.com/kastnerkyle/santa_barbaria
# See https://github.com/jych/cle for a library in this style
import numpy as np
from scipy import linalg
from scipy.io import loadmat
from functools import reduce
import numbers
import random
import theano
import zipfile
import gzip
import os
import glob
import sys
import subprocess
try:
    import cPickle as pickle
except ImportError:
    import pickle
from theano import tensor
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from collections import defaultdict


class sgd(object):
    """
    Vanilla SGD
    """
    def __init__(self, params):
        pass

    def updates(self, params, grads, learning_rate):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            updates.append((param, param - learning_rate * grad))
        return updates


class sgd_nesterov(object):
    """
    SGD with nesterov momentum

    Based on example from Yann D.
    """
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class rmsprop(object):
    """
    RMSProp with nesterov momentum and gradient rescaling
    """
    def __init__(self, params):
        self.running_square_ = [theano.shared(np.zeros_like(p.get_value()))
                                for p in params]
        self.running_avg_ = [theano.shared(np.zeros_like(p.get_value()))
                             for p in params]
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum, rescale=5.):
        grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), grads)))
        not_finite = tensor.or_(tensor.isnan(grad_norm),
                                tensor.isinf(grad_norm))
        grad_norm = tensor.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = tensor.maximum(rescale, grad_norm)
        # Magic constants
        combination_coeff = 0.9
        minimum_grad = 1E-4
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad = tensor.switch(not_finite, 0.1 * param,
                                 grad * (scaling_num / scaling_den))
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * tensor.sqr(grad)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * grad
            rms_grad = tensor.sqrt(new_square - new_avg ** 2)
            rms_grad = tensor.maximum(rms_grad, minimum_grad)
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad / rms_grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad / rms_grad
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class adagrad(object):
    """
    Adagrad optimizer
    """
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, eps=1E-8):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            m_t = memory + grad ** 2
            g_t = grad / (eps + tensor.sqrt(m_t))
            p_t = param - learning_rate * g_t
            updates.append((memory, m_t))
            updates.append((param, p_t))
        return updates


class adam(object):
    """
    Adam optimizer

    Based on implementation from @NewMu / Alex Radford
    """
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]
        self.velocity_ = [theano.shared(np.zeros_like(p.get_value()))
                          for p in params]
        self.itr_ = theano.shared(np.array(0.).astype(theano.config.floatX))

    def updates(self, params, grads, learning_rate, b1=0.1, b2=0.001, eps=1E-8):
        updates = []
        itr = self.itr_
        i_t = itr + 1.
        fix1 = 1. - (1. - b1) ** i_t
        fix2 = 1. - (1. - b2) ** i_t
        lr_t = learning_rate * (tensor.sqrt(fix2) / fix1)
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            velocity = self.velocity_[n]
            m_t = (b1 * grad) + ((1. - b1) * memory)
            v_t = (b2 * tensor.sqr(grad)) + ((1. - b2) * velocity)
            g_t = m_t / (tensor.sqrt(v_t) + eps)
            p_t = param - (lr_t * g_t)
            updates.append((memory, m_t))
            updates.append((velocity, v_t))
            updates.append((param, p_t))
        updates.append((itr, i_t))
        return updates


def get_dataset_dir(dataset_name, data_dir=None, folder=None, create_dir=True):
    """ Get dataset directory path """
    if not data_dir:
        data_dir = os.getenv("SANTA_BARBARIA_DATA", os.path.join(
            os.path.expanduser("~"), "santa_barbaria_data"))
    if folder is None:
        data_dir = os.path.join(data_dir, dataset_name)
    else:
        data_dir = os.path.join(data_dir, folder)
    if not os.path.exists(data_dir) and create_dir:
        os.makedirs(data_dir)
    return data_dir


def download(url, server_fname, local_fname=None, progress_update_percentage=5):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    try:
        import urllib
        urllib.urlretrieve('http://google.com')
    except AttributeError:
        import urllib.request as urllib
    u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def make_character_level_from_text(text):
    """ Create mapping and inverse mappings for text -> one_hot_char """
    all_chars = reduce(lambda x, y: set(x) | set(y), text, set())
    mapper = {k: n + 2 for n, k in enumerate(list(all_chars))}
    # 1 is EOS
    mapper["EOS"] = 1
    # 0 is UNK/MASK - unused here but needed in general
    mapper["UNK"] = 0
    inverse_mapper = {v: k for k, v in mapper.items()}

    def mapper_func(text_line):
        return [mapper[c] for c in text_line] + [mapper["EOS"]]

    def inverse_mapper_func(symbol_line):
        return "".join([inverse_mapper[s] for s in symbol_line
                        if s != mapper["EOS"]])

    # Remove blank lines
    cleaned = [mapper_func(t) for t in text if t != ""]
    return cleaned, mapper_func, inverse_mapper_func, mapper


def check_fetch_uci_words():
    """ Check for UCI vocabulary """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/'
    partial_path = get_dataset_dir("uci_words")
    full_path = os.path.join(partial_path, "uci_words.zip")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        # Download all 5 vocabularies and zip them into a file
        all_vocabs = ['vocab.enron.txt', 'vocab.kos.txt', 'vocab.nips.txt',
                      'vocab.nytimes.txt', 'vocab.pubmed.txt']
        for vocab in all_vocabs:
            dl_url = url + vocab
            download(dl_url, os.path.join(partial_path, vocab),
                     progress_update_percentage=1)

            def zipdir(path, zipf):
                # zipf is zipfile handle
                for root, dirs, files in os.walk(path):
                    for f in files:
                        if "vocab" in f:
                            zipf.write(os.path.join(root, f))

            zipf = zipfile.ZipFile(full_path, 'w')
            zipdir(partial_path, zipf)
            zipf.close()
    return full_path


def fetch_uci_words():
    """ Returns UCI vocabulary text. """
    data_path = check_fetch_uci_words()
    all_data = []
    with zipfile.ZipFile(data_path, "r") as f:
        for name in f.namelist():
            if ".txt" not in name:
                # Skip README
                continue
            data = f.read(name)
            data = data.split("\n")
            data = [l.strip() for l in data if l != ""]
            all_data.extend(data)
    return list(set(all_data))


def check_fetch_lovecraft():
    """ Check for lovecraft data """
    url = 'https://dl.dropboxusercontent.com/u/15378192/lovecraft_fiction.zip'
    partial_path = get_dataset_dir("lovecraft")
    full_path = os.path.join(partial_path, "lovecraft_fiction.zip")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_lovecraft():
    """ Returns lovecraft text. """
    data_path = check_fetch_lovecraft()
    all_data = []
    with zipfile.ZipFile(data_path, "r") as f:
        for name in f.namelist():
            if ".txt" not in name:
                # Skip README
                continue
            data = f.read(name)
            data = data.split("\n")
            data = [l.strip() for l in data if l != ""]
            all_data.extend(data)
    return all_data


def check_fetch_tfd():
    """ Check that tfd faces are downloaded """
    partial_path = get_dataset_dir("tfd")
    full_path = os.path.join(partial_path, "TFD_48x48.mat")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        raise ValueError("Put TFD_48x48 in %s" % str(partial_path))
    return full_path


def fetch_tfd():
    """ Returns flattened 48x48 TFD faces with pixel values in [0 - 1] """
    data_path = check_fetch_tfd()
    matfile = loadmat(data_path)
    all_data = matfile['images'].reshape(len(matfile['images']), -1) / 255.
    return all_data


def check_fetch_frey():
    """ Check that frey faces are downloaded """
    url = 'http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat'
    partial_path = get_dataset_dir("frey")
    full_path = os.path.join(partial_path, "frey_rawface.mat")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_frey():
    """ Returns flattened 20x28 frey faces with pixel values in [0 - 1] """
    data_path = check_fetch_frey()
    matfile = loadmat(data_path)
    all_data = (matfile['ff'] / 255.).T
    return all_data


def check_fetch_mnist():
    """ Check that mnist is downloaded. May need fixing for py3 compat """
    # py3k version is available at mnist_py3k.pkl.gz ... might need to fix
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    partial_path = get_dataset_dir("mnist")
    full_path = os.path.join(partial_path, "mnist.pkl.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_mnist():
    """ Returns mnist digits with pixel values in [0 - 1] """
    data_path = check_fetch_mnist()
    f = gzip.open(data_path, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()
    return train_set, valid_set, test_set


def check_fetch_binarized_mnist():
    raise ValueError("Binarized MNIST has no labels!")
    url = "https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz"
    partial_path = get_dataset_dir("binarized_mnist")
    fname = "binarized_mnist.npz"
    full_path = os.path.join(partial_path, fname)
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    """
    # Personal version
    url = "https://dl.dropboxusercontent.com/u/15378192/binarized_mnist_%s.npy"
    fname = "binarized_mnist_%s.npy"
    for s in ["train", "valid", "test"]:
        full_path = os.path.join(partial_path, fname % s)
        if not os.path.exists(partial_path):
            os.makedirs(partial_path)
        if not os.path.exists(full_path):
            download(url % s, full_path, progress_update_percentage=1)
    """
    return partial_path


def fetch_binarized_mnist():
    """ Get binarized version of MNIST data """
    train_set, valid_set, test_set = fetch_mnist()
    train_X = train_set[0]
    train_y = train_set[1]
    valid_X = valid_set[0]
    valid_y = valid_set[1]
    test_X = test_set[0]
    test_y = test_set[1]

    random_state = np.random.RandomState(1999)

    def get_sampled(arr):
        # make sure that a pixel can always be turned off
        return random_state.binomial(1, arr * 255 / 256., size=arr.shape)

    train_X = get_sampled(train_X)
    valid_X = get_sampled(valid_X)
    test_X = get_sampled(test_X)

    train_set = (train_X, train_y)
    valid_set = (valid_X, valid_y)
    test_set = (test_X, test_y)

    """
    # Old version for true binarized mnist
    data_path = check_fetch_binarized_mnist()
    fpath = os.path.join(data_path, "binarized_mnist.npz")

    arr = np.load(fpath)
    train_x = arr['train_data']
    valid_x = arr['valid_data']
    test_x = arr['test_data']
    train, valid, test = fetch_mnist()
    train_y = train[1]
    valid_y = valid[1]
    test_y = test[1]
    train_set = (train_x, train_y)
    valid_set = (valid_x, valid_y)
    test_set = (test_x, test_y)
    """
    return train_set, valid_set, test_set


def make_gif(arr, gif_name, plot_width, plot_height, list_text_per_frame=None,
             list_text_per_frame_color=None,
             delay=1, grayscale=False,
             loop=False, turn_on_agg=True):
    """ Make a gif frmo a series of pngs using matplotlib matshow """
    if turn_on_agg:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Plot temporaries for making gif
    # use random code to try and avoid deleting surprise files...
    random_code = random.randrange(2 ** 32)
    pre = str(random_code)
    for n, arr_i in enumerate(arr):
        plt.matshow(arr_i.reshape(plot_width, plot_height), cmap="gray")
        plt.axis('off')
        if list_text_per_frame is not None:
            text = list_text_per_frame[n]
            if list_text_per_frame_color is not None:
                color = list_text_per_frame_color[n]
            else:
                color = "white"
            plt.text(0, plot_height, text, color=color,
                     fontsize=2 * plot_height)
        # This looks rediculous but should count the number of digit places
        # also protects against multiple runs
        # plus 1 is to maintain proper ordering
        plotpath = '__%s_giftmp_%s.png' % (str(n).zfill(len(
            str(len(arr))) + 1), pre)
        plt.savefig(plotpath)
        plt.close()

    # make gif
    assert delay >= 1
    gif_delay = int(delay)
    basestr = "convert __*giftmp_%s.png -delay %s " % (pre, str(gif_delay))
    if loop:
        basestr += "-loop 1 "
    else:
        basestr += "-loop 0 "
    if grayscale:
        basestr += "-depth 8 -type Grayscale -depth 8 "
    basestr += "-resize %sx%s " % (str(int(5 * plot_width)),
                                   str(int(5 * plot_height)))
    basestr += gif_name
    print("Attempting gif")
    print(basestr)
    subprocess.call(basestr, shell=True)
    filelist = glob.glob("__*giftmp_%s.png" % pre)
    for f in filelist:
        os.remove(f)


def concatenate(tensor_list, name, axis=0, force_cast_to_float=True):
    """
    Wrapper to `theano.tensor.concatenate`.
    """
    if force_cast_to_float:
        tensor_list = cast_to_float(tensor_list)
    out = tensor.concatenate(tensor_list, axis=axis)
    conc_dim = int(sum([calc_expected_dim(inp)
                   for inp in tensor_list]))
    # This may be hosed... need to figure out how to generalize
    shape = list(expression_shape(tensor_list[0]))
    shape[axis] = conc_dim
    new_shape = tuple(shape)
    tag_expression(out, name, new_shape)
    return out


def theano_repeat(arr, n_repeat, stretch=False):
    """
    Create repeats of 2D array using broadcasting.
    Shape[0] incorrect after this node!
    """
    if arr.dtype not in ["float32", "float64"]:
        arr = tensor.cast(arr, "int32")
    if stretch:
        arg1 = arr.dimshuffle((0, 'x', 1))
        arg2 = tensor.alloc(1., 1, n_repeat, arr.shape[1])
        arg2 = tensor.cast(arg2, arr.dtype)
        cloned = (arg1 * arg2).reshape((n_repeat * arr.shape[0], arr.shape[1]))
    else:
        arg1 = arr.dimshuffle(('x', 0, 1))
        arg2 = tensor.alloc(1., n_repeat, 1, arr.shape[1])
        arg2 = tensor.cast(arg2, arr.dtype)
        cloned = (arg1 * arg2).reshape((n_repeat * arr.shape[0], arr.shape[1]))
    shape = expression_shape(arr)
    name = expression_name(arr)
    # Stretched shapes are *WRONG*
    tag_expression(cloned, name + "_stretched", (shape[0], shape[1]))
    return cloned


def cast_to_float(list_of_inputs):
    """ A cast that preserves name and shape info after cast """
    input_names = [inp.name for inp in list_of_inputs]
    cast_inputs = [tensor.cast(inp, theano.config.floatX)
                   for inp in list_of_inputs]
    for n, inp in enumerate(cast_inputs):
        cast_inputs[n].name = input_names[n]
    return cast_inputs


def interpolate_between_points(arr, n_steps=50):
    """ Helper function for drawing line between points in space """
    assert len(arr) > 2
    assert n_steps > 1
    path = [path_between_points(start, stop, n_steps=n_steps)
            for start, stop in zip(arr[:-1], arr[1:])]
    path = np.vstack(path)
    return path


def path_between_points(start, stop, n_steps=100, dtype=theano.config.floatX):
    """ Helper function for making a line between points in ND space """
    assert n_steps > 1
    step_vector = 1. / (n_steps - 1) * (stop - start)
    steps = np.arange(0, n_steps)[:, None] * np.ones((n_steps, len(stop)))
    steps = steps * step_vector + start
    return steps.astype(dtype)


def minibatch_indices(itr, minibatch_size):
    """ Generate indices for slicing 2D and 3D arrays in minibatches"""
    is_three_d = False
    if type(itr) is np.ndarray:
        if len(itr.shape) == 3:
            is_three_d = True
    elif not isinstance(itr[0], numbers.Real):
        # Assume 3D list of list of list
        # iterable of iterable of iterable, feature dim must be consistent
        is_three_d = True

    if is_three_d:
        if type(itr) is np.ndarray:
            minibatch_indices = np.arange(0, itr.shape[1], minibatch_size)
        else:
            # multi-list
            minibatch_indices = np.arange(0, len(itr), minibatch_size)
        minibatch_indices = np.asarray(list(minibatch_indices) + [len(itr)])
        start_indices = minibatch_indices[:-1]
        end_indices = minibatch_indices[1:]
        return zip(start_indices, end_indices)
    else:
        minibatch_indices = np.arange(0, len(itr), minibatch_size)
        minibatch_indices = np.asarray(list(minibatch_indices) + [len(itr)])
        start_indices = minibatch_indices[:-1]
        end_indices = minibatch_indices[1:]
        return zip(start_indices, end_indices)


def convert_to_one_hot(itr, n_classes, dtype="int32"):
    """ Convert 2D or 3D iterators to one_hot. Primarily for text. """
    is_three_d = False
    if type(itr) is np.ndarray:
        if len(itr.shape) == 3:
            is_three_d = True
    elif not isinstance(itr[0], numbers.Real):
        # Assume 3D list of list of list
        # iterable of iterable of iterable, feature dim must be consistent
        is_three_d = True

    if is_three_d:
        lengths = [len(i) for i in itr]
        one_hot = np.zeros((max(lengths), len(itr), n_classes), dtype=dtype)
        for n in range(len(itr)):
            one_hot[np.arange(lengths[n]), n, itr[n]] = 1
    else:
        one_hot = np.zeros((len(itr), n_classes), dtype=dtype)
        one_hot[np.arange(len(itr)), itr] = 1
    return one_hot


def save_checkpoint(save_path, items_dict):
    """ Simple wrapper for checkpoint dictionaries """
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="wb") as f:
        pickle.dump(items_dict, f)
    sys.setrecursionlimit(old_recursion_limit)


def load_checkpoint(save_path):
    """ Simple pickle wrapper for checkpoint dictionaries """
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="rb") as f:
        items_dict = pickle.load(f)
    sys.setrecursionlimit(old_recursion_limit)
    return items_dict


def print_status_func(epoch_results):
    """ Print the last results from a results dictionary """
    n_epochs_seen = max([len(l) for l in epoch_results.values()])
    last_results = {k: v[-1] for k, v in epoch_results.items()}
    print("Epoch %i: %s" % (n_epochs_seen, last_results))


def checkpoint_status_func(save_path, checkpoint_dict, epoch_results):
    """ Saves a checkpoint dict """
    checkpoint_dict["previous_epoch_results"] = epoch_results
    save_checkpoint(save_path, checkpoint_dict)
    print_status_func(epoch_results)


def early_stopping_status_func(valid_cost, save_path, checkpoint_dict,
                               epoch_results):
    """
    Adds valid_cost to epoch_results and saves model if best valid
    Assumes checkpoint_dict is a defaultdict(list)

    Example usage for early stopping on validation set:

    def status_func(status_number, epoch_number, epoch_results):
        valid_results = iterate_function(
            cost_function, [X_clean_valid, y_clean_valid], minibatch_size,
            list_of_output_names=["valid_cost"],
            list_of_minibatch_functions=[text_minibatcher], n_epochs=1,
            shuffle=False)
        early_stopping_status_func(valid_results["valid_cost"][-1],
                                save_path, checkpoint_dict, epoch_results)

    status_func can then be fed to iterate_function for training with early
    stopping.
    """
    # Quick trick to avoid 0 length list
    old = min(epoch_results["valid_cost"] + [np.inf])
    epoch_results["valid_cost"].append(valid_cost)
    new = min(epoch_results["valid_cost"])
    if new < old:
        print("Saving checkpoint based on validation score")
        checkpoint_status_func(save_path, checkpoint_dict, epoch_results)
    else:
        print_status_func(epoch_results)


def even_slice(arr, size):
    """ Force array to be even by slicing off the end """
    extent = -(len(arr) % size)
    if extent == 0:
        extent = None
    return arr[:extent]


def make_minibatch(arg, start, stop):
    """ Does not handle off-size minibatches """
    if len(arg.shape) == 3:
        return [arg[:, start:stop]]
    else:
        return [arg[start:stop]]


def gen_text_minibatch_func(one_hot_size):
    """
    Returns a function that will turn a text minibatch into one_hot form.

    For use with iterate_function list_of_minibatch_functions argument.

    Example:
    n_chars = 84
    text_minibatcher = gen_text_minibatch_func(n_chars)
    valid_results = iterate_function(
        cost_function, [X_clean_valid, y_clean_valid], minibatch_size,
        list_of_output_names=["valid_cost"],
        list_of_minibatch_functions=[text_minibatcher], n_epochs=1,
        shuffle=False)
    """
    def apply(arg, start, stop):
        sli = arg[start:stop]
        expanded = convert_to_one_hot(sli, one_hot_size)
        lengths = [len(s) for s in sli]
        mask = np.zeros((max(lengths), len(sli)), dtype=theano.config.floatX)
        for n, l in enumerate(lengths):
            mask[np.arange(l), n] = 1.
        return expanded, mask
    return apply


def iterate_function(func, list_of_minibatch_args, minibatch_size,
                     list_of_non_minibatch_args=None,
                     list_of_minibatch_functions=[make_minibatch],
                     list_of_output_names=None,
                     n_epochs=1000, n_status=50, status_func=None,
                     previous_epoch_results=None,
                     shuffle=False, random_state=None):
    """
    Minibatch arguments should come first.

    Constant arguments which should not be iterated can be passed as
    list_of_non_minibatch_args.

    If list_of_minbatch_functions is length 1, will be replicated to length of
    list_of_args - applying the same function to all minibatch arguments in
    list_of_args. Otherwise, this should be the same length as list_of_args

    list_of_output_names simply names the output of the passed in function.
    Should be the same length as the number of outputs from the function.

    status_func is a function run periodically (based on n_status_points),
    which allows for validation, early stopping, checkpointing, etc.

    previous_epoch_results allows for continuing from saved checkpoints

    shuffle and random_state are used to determine if minibatches are run
    in sequence or selected randomly each epoch.

    By far the craziest function in this file.

    Example validation function:
    n_chars = 84
    text_minibatcher = gen_text_minibatch_func(n_chars)

    cost_function returns one value, the cost for that minibatch

    valid_results = iterate_function(
        cost_function, [X_clean_valid, y_clean_valid], minibatch_size,
        list_of_output_names=["valid_cost"],
        list_of_minibatch_functions=[text_minibatcher], n_epochs=1,
        shuffle=False)

    Example training loop:

    fit_function returns 3 values, nll, kl and the total cost

    epoch_results = iterate_function(fit_function, [X, y], minibatch_size,
                                 list_of_output_names=["nll", "kl", "cost"],
                                 n_epochs=2000,
                                 status_func=status_func,
                                 previous_epoch_results=previous_epoch_results,
                                 shuffle=True,
                                 random_state=random_state)
    """
    if previous_epoch_results is None:
        epoch_results = defaultdict(list)
    else:
        epoch_results = previous_epoch_results
    # Input checking and setup
    if shuffle:
        assert random_state is not None
    status_points = list(range(n_epochs))
    if len(status_points) >= n_status:
        intermediate_points = status_points[::n_epochs // n_status]
        status_points = intermediate_points + [status_points[-1]]
    else:
        status_points = range(len(status_points))

    for arg in list_of_minibatch_args:
        assert len(arg) == len(list_of_minibatch_args[0])

    indices = minibatch_indices(list_of_minibatch_args[0], minibatch_size)
    if len(list_of_minibatch_args[0]) % minibatch_size != 0:
        print ("length of dataset should be evenly divisible by "
               "minibatch_size.")
    if len(list_of_minibatch_functions) == 1:
        list_of_minibatch_functions = list_of_minibatch_functions * len(
            list_of_minibatch_args)
    else:
        assert len(list_of_minibatch_functions) == len(list_of_minibatch_args)
    # Function loop
    for e in range(n_epochs):
        results = defaultdict(list)
        if shuffle:
            random_state.shuffle(indices)
        for i, j in indices:
            minibatch_args = []
            for n, arg in enumerate(list_of_minibatch_args):
                minibatch_args += list_of_minibatch_functions[n](arg, i, j)
            if list_of_non_minibatch_args is not None:
                all_args = minibatch_args + list_of_non_minibatch_args
            else:
                all_args = minibatch_args
            minibatch_results = func(*all_args)
            if type(minibatch_results) is not list:
                minibatch_results = [minibatch_results]
            for n, k in enumerate(minibatch_results):
                if list_of_output_names is not None:
                    assert len(list_of_output_names) == len(minibatch_results)
                    results[list_of_output_names[n]].append(
                        minibatch_results[n])
                else:
                    results[n].append(minibatch_results[n])
        avg_output = {r: np.mean(results[r]) for r in results.keys()}
        for k in avg_output.keys():
            epoch_results[k].append(avg_output[k])
        if e in status_points:
            if status_func is not None:
                epoch_number = e
                status_number = np.searchsorted(status_points, e)
                status_func(status_number, epoch_number, epoch_results)
    return epoch_results


def as_shared(arr, name=None):
    """ Quick wrapper for theano.shared """
    if name is not None:
        return theano.shared(value=arr, borrow=True)
    else:
        return theano.shared(value=arr, name=name, borrow=True)


def np_zeros(shape):
    """ Builds a numpy variable filled with zeros """
    return np.zeros(shape).astype(theano.config.floatX)


def np_rand(shape, random_state):
    # Make sure bounds aren't the same
    return random_state.uniform(low=-0.08, high=0.08, size=shape).astype(
        theano.config.floatX)


def np_randn(shape, random_state):
    """ Builds a numpy variable filled with random normal values """
    return (0.01 * random_state.randn(*shape)).astype(theano.config.floatX)


def np_tanh_fan(shape, random_state):
    # The . after the 6 is critical! shape has dtype int...
    bound = np.sqrt(6. / np.sum(shape))
    return random_state.uniform(low=-bound, high=bound,
                                size=shape).astype(theano.config.floatX)


def np_sigmoid_fan(shape, random_state):
    return 4 * np_tanh_fan(shape, random_state)


def np_ortho(shape, random_state):
    """ Builds a theano variable filled with orthonormal random values """
    g = random_state.randn(*shape)
    o_g = linalg.svd(g)[0]
    return o_g.astype(theano.config.floatX)


def names_in_graph(list_of_names, graph):
    """ Return true if all names are in the graph """
    return all([name in graph.keys() for name in list_of_names])


def add_arrays_to_graph(list_of_arrays, list_of_names, graph, strict=True):
    assert len(list_of_arrays) == len(list_of_names)
    arrays_added = []
    for array, name in zip(list_of_arrays, list_of_names):
        if name in graph.keys() and strict:
            raise ValueError("Name %s already found in graph!" % name)
        shared_array = as_shared(array, name=name)
        graph[name] = shared_array
        arrays_added.append(shared_array)


def make_shapename(name, shape):
    if len(shape) == 1:
        # vector, primarily init hidden state for RNN
        return name + "_kdl_" + str(shape[0]) + "x"
    else:
        return name + "_kdl_" + "x".join(map(str, list(shape)))


def parse_shapename(shapename):
    try:
        # Bracket for scan
        shape = shapename.split("_kdl_")[1].split("[")[0].split("x")
    except AttributeError:
        raise AttributeError("Unable to parse shapename. Has the expression "
                             "been tagged with a shape by tag_expression? "
                             " input shapename was %s" % shapename)
    if "[" in shapename.split("_kdl_")[1]:
        # inside scan
        shape = shape[1:]
    name = shapename.split("_kdl_")[0]
    # More cleaning to handle scan
    shape = tuple([int(s) for s in shape if s != ''])
    return name, shape


def add_datasets_to_graph(list_of_datasets, list_of_names, graph, strict=True,
                          list_of_test_values=None):
    assert len(list_of_datasets) == len(list_of_names)
    datasets_added = []
    for n, (dataset, name) in enumerate(zip(list_of_datasets, list_of_names)):
        if dataset.dtype != "int32":
            if len(dataset.shape) == 1:
                sym = tensor.vector()
            elif len(dataset.shape) == 2:
                sym = tensor.matrix()
            elif len(dataset.shape) == 3:
                sym = tensor.tensor3()
            else:
                raise ValueError("dataset %s has unsupported shape" % name)
        elif dataset.dtype == "int32":
            if len(dataset.shape) == 1:
                sym = tensor.ivector()
            elif len(dataset.shape) == 2:
                sym = tensor.imatrix()
            elif len(dataset.shape) == 3:
                sym = tensor.itensor3()
            else:
                raise ValueError("dataset %s has unsupported shape" % name)
        else:
            raise ValueError("dataset %s has unsupported dtype %s" % (
                name, dataset.dtype))
        if list_of_test_values is not None:
            sym.tag.test_value = list_of_test_values[n]
        tag_expression(sym, name, dataset.shape)
        datasets_added.append(sym)
    graph["__datasets_added__"] = datasets_added
    return datasets_added


def tag_expression(expression, name, shape):
    expression.name = make_shapename(name, shape)


def expression_name(expression):
    return parse_shapename(expression.name)[0]


def expression_shape(expression):
    return parse_shapename(expression.name)[1]


def calc_expected_dim(expression):
    # super intertwined with add_datasets_to_graph
    # Expect variables representing datasets in graph!!!
    # Function graph madness
    # Shape format is HxWxZ
    shape = expression_shape(expression)
    dim = shape[-1]
    return dim


def fetch_from_graph(list_of_names, graph):
    """ Returns a list of shared variables from the graph """
    if "__datasets_added__" not in graph.keys():
        # Check for dataset in graph
        raise AttributeError("No dataset in graph! Make sure to add "
                             "the dataset using add_datasets_to_graph")
    return [graph[name] for name in list_of_names]


def get_params_and_grads(graph, cost):
    grads = []
    params = []
    for k, p in graph.items():
        if k[:2] == "__":
            # skip private tags
            continue
        print("Computing grad w.r.t %s" % k)
        grad = tensor.grad(cost, p)
        params.append(p)
        grads.append(grad)
    return params, grads


def binary_crossentropy_nll(predicted_values, true_values):
    """ Returns likelihood compared to binary true_values """
    return (-true_values * tensor.log(predicted_values) - (
        1 - true_values) * tensor.log(1 - predicted_values)).sum(axis=-1)


def binary_entropy(values):
    return (-values * tensor.log(values)).sum(axis=-1)


def categorical_crossentropy_nll(predicted_values, true_values):
    """ Returns likelihood compared to one hot category labels """
    indices = tensor.argmax(true_values, axis=-1)
    rows = tensor.arange(true_values.shape[0])
    if predicted_values.ndim < 3:
        return -tensor.log(predicted_values)[rows, indices]
    elif predicted_values.ndim == 3:
        d0 = true_values.shape[0]
        d1 = true_values.shape[1]
        pred = predicted_values.reshape((d0 * d1, -1))
        ind = indices.reshape((d0 * d1,))
        s = tensor.arange(pred.shape[0])
        correct = -tensor.log(pred)[s, ind]
        return correct.reshape((d0, d1,))
    else:
        raise AttributeError("Tensor dim not supported")


def abs_error_nll(predicted_values, true_values):
    return tensor.abs_(predicted_values - true_values).sum(axis=-1)


def squared_error_nll(predicted_values, true_values):
    return tensor.sqr(predicted_values - true_values).sum(axis=-1)


def gaussian_error_nll(mu_values, sigma_values, true_values):
    """ sigma should come from a softplus layer """
    nll = 0.5 * (mu_values - true_values) ** 2 / sigma_values ** 2 + tensor.log(
        2 * np.pi * sigma_values ** 2)
    return nll


def log_gaussian_error_nll(mu_values, log_sigma_values, true_values):
    """ log_sigma should come from a linear layer """
    nll = 0.5 * (mu_values - true_values) ** 2 / tensor.exp(
        log_sigma_values) ** 2 + tensor.log(2 * np.pi) + 2 * log_sigma_values
    return nll


def masked_cost(cost, mask):
    return cost * mask


def softplus(X):
    return tensor.nnet.softplus(X) + 1E-4


def relu(X):
    return X * (X > 1)


def linear(X):
    return X


def softmax(X):
    # should work for both 2D and 3D
    e_X = tensor.exp(X - X.max(axis=-1, keepdims=True))
    out = e_X / e_X.sum(axis=-1, keepdims=True)
    return out


def dropout(X, random_state, on_off_switch, p=0.):
    if p > 0:
        theano_seed = random_state.randint(-2147462579, 2147462579)
        # Super edge case...
        if theano_seed == 0:
            print("WARNING: prior layer got 0 seed. Reseeding...")
            theano_seed = random_state.randint(-2**32, 2**32)
        theano_rng = MRG_RandomStreams(seed=theano_seed)
        retain_prob = 1 - p
        if X.ndim == 2:
            X *= theano_rng.binomial(
                X.shape, p=retain_prob,
                dtype=theano.config.floatX) ** on_off_switch
            X /= retain_prob
        elif X.ndim == 3:
            # Dropout for recurrent - don't drop over time!
            X *= theano_rng.binomial((
                X.shape[1], X.shape[2]), p=retain_prob,
                dtype=theano.config.floatX) ** on_off_switch
            X /= retain_prob
        else:
            raise ValueError("Unsupported tensor with ndim %s" % str(X.ndim))
    return X


def dropout_layer(list_of_inputs, name, on_off_switch, dropout_prob=0.5,
                  random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    conc_input = concatenate(list_of_inputs, name, axis=-1)
    shape = expression_shape(conc_input)
    dropped = dropout(conc_input, random_state, on_off_switch, p=dropout_prob)
    tag_expression(dropped, name, shape)
    return dropped


def projection_layer(list_of_inputs, graph, name, proj_dim=None,
                     random_state=None, strict=True, init_func=np_tanh_fan,
                     func=linear):
    W_name = name + '_W'
    b_name = name + '_b'
    list_of_names = [W_name, b_name]
    if not names_in_graph(list_of_names, graph):
        assert proj_dim is not None
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dim(inp)
                                  for inp in list_of_inputs]))
        np_W = init_func((conc_input_dim, proj_dim), random_state)
        np_b = np_zeros((proj_dim,))
        add_arrays_to_graph([np_W, np_b], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)
    W, b = fetch_from_graph(list_of_names, graph)
    conc_input = concatenate(list_of_inputs, name, axis=-1)
    output = tensor.dot(conc_input, W) + b
    if func is not None:
        final = func(output)
    else:
        final = output
    shape = list(expression_shape(conc_input))
    # Projection is on last axis
    shape[-1] = proj_dim
    new_shape = tuple(shape)
    tag_expression(final, name, new_shape)
    return final


def linear_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
                 strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=linear)


def sigmoid_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
                  strict=True, init_func=np_sigmoid_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=tensor.nnet.sigmoid)


def tanh_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
               strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=tensor.tanh)


def softplus_layer(list_of_inputs, graph, name, proj_dim=None,
                   random_state=None, strict=True,
                   init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=softplus)


def exp_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
              strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=tensor.exp)


def relu_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
               strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=relu)


def softmax_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
                  strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=softmax)


def softmax_sample_layer(list_of_multinomial_inputs, name, random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_multinomial = concatenate(list_of_multinomial_inputs, name, axis=1)
    shape = expression_shape(conc_multinomial)
    conc_multinomial /= len(list_of_multinomial_inputs)
    tag_expression(conc_multinomial, name, shape)
    samp = theano_rng.multinomial(pvals=conc_multinomial,
                                  dtype="int32")
    tag_expression(samp, name, (shape[0], shape[1]))
    return samp


def gaussian_sample_layer(list_of_mu_inputs, list_of_sigma_inputs,
                          name, random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_mu = concatenate(list_of_mu_inputs, name, axis=1)
    conc_sigma = concatenate(list_of_sigma_inputs, name, axis=1)
    e = theano_rng.normal(size=(conc_sigma.shape[0],
                                conc_sigma.shape[1]),
                          dtype=conc_sigma.dtype)
    samp = conc_mu + conc_sigma * e
    shape = expression_shape(conc_sigma)
    tag_expression(samp, name, shape)
    return samp


def gaussian_log_sample_layer(list_of_mu_inputs, list_of_log_sigma_inputs,
                              name, random_state=None):
    """ log_sigma_inputs should be from a linear_layer """
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_mu = concatenate(list_of_mu_inputs, name, axis=1)
    conc_log_sigma = concatenate(list_of_log_sigma_inputs, name, axis=1)
    e = theano_rng.normal(size=(conc_log_sigma.shape[0],
                                conc_log_sigma.shape[1]),
                          dtype=conc_log_sigma.dtype)

    samp = conc_mu + tensor.exp(0.5 * conc_log_sigma) * e
    shape = expression_shape(conc_log_sigma)
    tag_expression(samp, name, shape)
    return samp


def gaussian_kl(list_of_mu_inputs, list_of_sigma_inputs, name):
    conc_mu = concatenate(list_of_mu_inputs, name)
    conc_sigma = concatenate(list_of_sigma_inputs, name)
    kl = 0.5 * tensor.sum(-2 * tensor.log(conc_sigma) + conc_mu ** 2
                          + conc_sigma ** 2 - 1, axis=1)
    return kl


def gaussian_log_kl(list_of_mu_inputs, list_of_log_sigma_inputs, name):
    """ log_sigma_inputs should come from linear layer"""
    conc_mu = concatenate(list_of_mu_inputs, name)
    conc_log_sigma = 0.5 * concatenate(list_of_log_sigma_inputs, name)
    kl = 0.5 * tensor.sum(-2 * conc_log_sigma + conc_mu ** 2
                          + tensor.exp(conc_log_sigma) ** 2 - 1, axis=1)
    return kl


def switch_wrap(switch_func, if_true_var, if_false_var, name):
    switched = tensor.switch(switch_func, if_true_var, if_false_var)
    shape = expression_shape(if_true_var)
    assert shape == expression_shape(if_false_var)
    tag_expression(switched, name, shape)
    return switched
