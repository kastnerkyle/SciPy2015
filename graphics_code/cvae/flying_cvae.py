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
predict_function = checkpoint_dict["predict_function"]

random_state = np.random.RandomState(args.seed)
train, valid, test = fetch_binarized_mnist()
# visualize against validation so we aren't cheating
X = valid[0].astype(theano.config.floatX)
y = valid[1].astype("int32")

# number of samples
n_plot_samples = 6
n_classes = 10
# MNIST dimensions
width = 28
height = 28
# Get random data samples
ind = np.arange(len(X))
random_state.shuffle(ind)
sample_X = X[ind[:n_plot_samples]]
sample_y = convert_to_one_hot(y[ind[:n_plot_samples]], n_classes=n_classes)


def gen_samples(X, y):
    mu, log_sig = encode_function(X, y)
    # No noise at test time - repeat y twice because y_pred is needed for Theano
    # But it is not used unless y_sym is all -1
    out, = decode_function(mu + np.exp(log_sig), y)
    return out

# VAE specific plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

all_pred_y, = predict_function(X)
all_pred_y = np.argmax(all_pred_y, axis=1)
accuracy = np.mean(all_pred_y.ravel() == y.ravel())

f, axarr = plt.subplots(n_plot_samples, 2)
n_correct_to_show = n_plot_samples // 2
n_incorrect_to_show = n_plot_samples - n_correct_to_show

correct_ind = np.where(all_pred_y == y)[0]
incorrect_ind = np.where(all_pred_y != y)[0]
random_state.shuffle(correct_ind)
random_state.shuffle(incorrect_ind)
c = correct_ind[:n_correct_to_show]
i = incorrect_ind[:n_incorrect_to_show]

X_corr = X[c]
X_incorr = X[i]
X_stack = np.vstack((X_corr, X_incorr))
y_corr = convert_to_one_hot(y[c], n_classes=10)
y_incorr = convert_to_one_hot(y[i], n_classes=10)
y_stack = np.vstack((y_corr, y_incorr))

generated_X = gen_samples(X_stack, y_stack)
predicted_y = convert_to_one_hot(np.hstack((all_pred_y[c], all_pred_y[i])),
                                 n_classes=10)

for n, (X_i, y_i, sx_i, sy_i) in enumerate(zip(X_stack, y_stack,
                                               generated_X, predicted_y)):
    axarr[n, 0].matshow(X_i.reshape(width, height), cmap="gray")
    axarr[n, 1].matshow(sx_i.reshape(width, height), cmap="gray")
    axarr[n, 0].axis('off')
    axarr[n, 1].axis('off')

    y_a = np.argmax(y_i)
    sy_a = np.argmax(sy_i)
    axarr[n, 0].text(0, 7, str(y_a), color='green')
    if y_a == sy_a:
        axarr[n, 1].text(0, 7, str(sy_a), color='green')
    else:
        axarr[n, 1].text(0, 7, str(sy_a), color='red')

f.suptitle("Validation accuracy: %s" % str(accuracy))
plt.savefig('vae_reconstruction.png')
plt.close()

# Style plotting
f, axarr = plt.subplots(n_plot_samples, n_classes + 1)
for n, (X_i, y_i) in enumerate(zip(sample_X, sample_y)):
    true_rec = gen_samples(X_i[None], y_i[None])
    fixed_mu, fixed_sigma = encode_function(X_i[None], y_i[None])
    axarr[n, 0].matshow(X_i.reshape(width, height), cmap="gray")
    axarr[n, 0].axis('off')
    all_mu = fixed_mu * np.ones((n_classes, fixed_mu.shape[1])).astype(
        "float32")
    all_sigma = fixed_sigma * np.ones((n_classes, fixed_sigma.shape[1])).astype(
        "float32")
    all_classes = np.eye(n_classes).astype('int32')
    all_recs, = decode_function(all_mu + np.exp(all_sigma), all_classes)
    for j in range(1, n_classes + 1):
        axarr[n, j].matshow(all_recs[j - 1].reshape(width, height), cmap="gray")
        axarr[n, j].axis('off')
f.suptitle("Style variation by changing conditional")
plt.savefig('vae_style.png')
plt.close()

# Calculate noisy linear path between points in space
mus, log_sigmas = encode_function(sample_X, sample_y)
n_steps = 20
mu_path = interpolate_between_points(mus, n_steps=n_steps)
log_sigma_path = interpolate_between_points(log_sigmas, n_steps=n_steps)

# Noisy path across space from one point to another
path_X = mu_path + np.exp(log_sigma_path)
path_y = np.zeros((len(path_X), n_classes), dtype="int32")

for i in range(n_plot_samples):
    path_y[i * n_steps:(i + 1) * n_steps] = sample_y[i]

# Have to pass another argument for y_pred
# But it is not used unless y_sym is all -1
out, = decode_function(path_X, path_y)
text_y = [str(np.argmax(path_y[i])) for i in range(len(path_y))]
color_y = ["white"] * len(text_y)
make_gif(out, "vae_code.gif", width, height, list_text_per_frame=text_y,
         list_text_per_frame_color=color_y, delay=1, grayscale=True)
