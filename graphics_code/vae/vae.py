from kdl_template import *

train, valid, test = fetch_binarized_mnist()
X = train[0].astype(theano.config.floatX)

# graph holds information necessary to build layers from parents
graph = OrderedDict()
X_sym, = add_datasets_to_graph([X], ["X"], graph)
# random state so script is deterministic
random_state = np.random.RandomState(1999)

minibatch_size = 100
n_code = 100
n_enc_layer = [200, 200]
n_dec_layer = [200, 200]
width = 28
height = 28
n_input = width * height

# encode path aka q
l1_enc = softplus_layer([X_sym], graph, 'l1_enc', n_enc_layer[0], random_state)
l2_enc = softplus_layer([l1_enc], graph, 'l2_enc',  n_enc_layer[1],
                        random_state)
code_mu = linear_layer([l2_enc], graph, 'code_mu', n_code, random_state)
code_log_sigma = linear_layer([l2_enc], graph, 'code_log_sigma', n_code,
                              random_state)
kl = gaussian_log_kl([code_mu], [code_log_sigma], 'kl').mean()
samp = gaussian_log_sample_layer([code_mu], [code_log_sigma], 'samp',
                                 random_state)

# decode path aka p
l1_dec = softplus_layer([samp], graph, 'l1_dec',  n_dec_layer[0], random_state)
l2_dec = softplus_layer([l1_dec], graph, 'l2_dec', n_dec_layer[1], random_state)
out = sigmoid_layer([l2_dec], graph, 'out', n_input, random_state)

nll = binary_crossentropy_nll(out, X_sym).mean()
# log p(x) = -nll so swap sign
# want to minimize cost in optimization so multiply by -1
cost = -1 * (-nll - kl)
params, grads = get_params_and_grads(graph, cost)

learning_rate = 0.0003
opt = adam(params)
updates = opt.updates(params, grads, learning_rate)

# Checkpointing
save_path = "serialized_vae.pkl"
if not os.path.exists(save_path):
    fit_function = theano.function([X_sym], [nll, kl, nll + kl],
                                   updates=updates)
    encode_function = theano.function([X_sym], [code_mu, code_log_sigma])
    decode_function = theano.function([samp], [out])
    checkpoint_dict = {}
    checkpoint_dict["fit_function"] = fit_function
    checkpoint_dict["encode_function"] = encode_function
    checkpoint_dict["decode_function"] = decode_function
    previous_epoch_results = None
else:
    checkpoint_dict = load_checkpoint(save_path)
    fit_function = checkpoint_dict["fit_function"]
    encode_function = checkpoint_dict["encode_function"]
    decode_function = checkpoint_dict["decode_function"]
    previous_epoch_results = checkpoint_dict["previous_epoch_results"]


def status_func(status_number, epoch_number, epoch_results):
    checkpoint_status_func(save_path, checkpoint_dict, epoch_results)

epoch_results = iterate_function(fit_function, [X], minibatch_size,
                                 list_of_output_names=["nll", "kl",
                                                       "lower_bound"],
                                 n_epochs=5000,
                                 status_func=status_func,
                                 previous_epoch_results=previous_epoch_results,
                                 shuffle=True,
                                 random_state=random_state)
