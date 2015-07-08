from kdl_template import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("saved_functions_file",
                    help="Saved pickle file from vae training")
parser.add_argument("--seed", "-s",
                    help="random seed for path calculation",
                    action="store", default=1979, type=int)

args = parser.parse_args()
if not os.path.exists(args.saved_functions_file):
    raise ValueError("Please provide a valid path for saved pickle file!")

checkpoint_dict = load_checkpoint(args.saved_functions_file)
encode_function = checkpoint_dict["encode_function"]
decode_function = checkpoint_dict["decode_function"]

random_state = np.random.RandomState(args.seed)
train, valid, test = fetch_binarized_mnist()
# visualize against validation so we aren't cheating
X = valid[0].astype(theano.config.floatX)

# number of samples
n_plot_samples = 5
# MNIST dimensions
width = 28
height = 28
# Get random data samples
ind = np.arange(len(X))
random_state.shuffle(ind)
sample_X = X[ind[:n_plot_samples]]


def gen_samples(arr):
    mu, log_sig = encode_function(arr)
    # No noise at test time
    out, = decode_function(mu + np.exp(log_sig))
    return out

# VAE specific plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
samples = gen_samples(sample_X)
f, axarr = plt.subplots(n_plot_samples, 2)
for n, (X_i, s_i) in enumerate(zip(sample_X, samples)):
    axarr[n, 0].matshow(X_i.reshape(width, height), cmap="gray")
    axarr[n, 1].matshow(s_i.reshape(width, height), cmap="gray")
    axarr[n, 0].axis('off')
    axarr[n, 1].axis('off')
plt.savefig('vae_reconstruction.png')
plt.close()

# Calculate linear path between points in space
mus, log_sigmas = encode_function(sample_X)
mu_path = interpolate_between_points(mus)
log_sigma_path = interpolate_between_points(log_sigmas)

# Path across space from one point to another
path = mu_path + np.exp(log_sigma_path)
out, = decode_function(path)
make_gif(out, "vae_code.gif", width, height, delay=1, grayscale=True)
