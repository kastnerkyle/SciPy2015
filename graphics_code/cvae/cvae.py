from kdl_template import *

train, valid, test = fetch_binarized_mnist()
X = train[0].astype(theano.config.floatX)
y = convert_to_one_hot(train[1], n_classes=10)

# graph holds information necessary to build layers from parents
graph = OrderedDict()
X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
# random state so script is deterministic
random_state = np.random.RandomState(1999)

minibatch_size = 100
n_code = 100
n_targets = 10
n_enc_layer = [200, 200, 200, 200]
n_dec_layer = [200, 200, 200]
width = 28
height = 28
n_input = width * height

# q(y | x)
y_l1_enc = softplus_layer([X_sym], graph, 'y_l1_enc', n_enc_layer[0],
                          random_state)
y_l2_enc = softmax_layer([y_l1_enc], graph, 'y_l2_enc', n_enc_layer[1],
                         random_state)
y_pred = softmax_layer([y_l2_enc], graph, 'y_pred_enc', n_targets,
                       random_state)

# partial q(z | x)
x_l1_enc = softplus_layer([X_sym], graph, 'x_l1_enc', n_enc_layer[0],
                          random_state)
x_l2_enc = softplus_layer([x_l1_enc], graph, 'x_l2_enc',  n_enc_layer[1],
                          random_state)


# combined q(y | x) and partial q(z | x) for q(z | y, x)
l3_enc = softplus_layer([x_l2_enc, y_pred], graph, 'l3_enc', n_enc_layer[2],
                        random_state)
l4_enc = softplus_layer([l3_enc], graph, 'l4_enc', n_enc_layer[3],
                        random_state)
code_mu = linear_layer([l4_enc], graph, 'code_mu', n_code, random_state)
code_log_sigma = linear_layer([l4_enc], graph, 'code_log_sigma', n_code,
                              random_state)
samp = gaussian_log_sample_layer([code_mu], [code_log_sigma], 'samp',
                                 random_state)

# decode path aka p for labeled data
l1_dec = softplus_layer([samp, y_sym], graph, 'l1_dec',  n_dec_layer[0],
                        random_state)
l2_dec = softplus_layer([l1_dec], graph, 'l2_dec', n_dec_layer[1], random_state)
l3_dec = softplus_layer([l2_dec], graph, 'l3_dec', n_dec_layer[2], random_state)
out = sigmoid_layer([l3_dec], graph, 'out', n_input, random_state)

# Components of the cost
nll = binary_crossentropy_nll(out, X_sym).mean()
ent = binary_entropy(y_pred).mean()
kl = gaussian_log_kl([code_mu], [code_log_sigma], 'kl').mean()

# Junk when in unlabled mode
err = categorical_crossentropy_nll(y_pred, y_sym).mean()

# log p(x) = -nll so swap sign
# want to minimize cost in optimization so multiply by -1
base_cost = -1 * (-nll - kl)

# -log q(y | x) is nll already
alpha = .1
cost = base_cost + alpha * err

params, grads = get_params_and_grads(graph, cost)
learning_rate = 0.0003
opt = adam(params)
updates = opt.updates(params, grads, learning_rate)

# Checkpointing
save_path = "serialized_cvae.pkl"
if not os.path.exists(save_path):
    fit_function = theano.function([X_sym, y_sym], [nll, kl, nll + kl],
                                   updates=updates)
    predict_function = theano.function([X_sym], [y_pred])
    encode_function = theano.function([X_sym, y_sym], [code_mu, code_log_sigma],
                                      on_unused_input='warn')
    # Need both due to tensor.switch, but only one should ever be used
    decode_function = theano.function([samp, y_sym], [out])
    checkpoint_dict = {}
    checkpoint_dict["fit_function"] = fit_function
    checkpoint_dict["predict_function"] = predict_function
    checkpoint_dict["encode_function"] = encode_function
    checkpoint_dict["decode_function"] = decode_function
    previous_epoch_results = None
else:
    checkpoint_dict = load_checkpoint(save_path)
    fit_function = checkpoint_dict["fit_function"]
    predict_function = checkpoint_dict["predict_function"]
    encode_function = checkpoint_dict["encode_function"]
    decode_function = checkpoint_dict["decode_function"]
    previous_epoch_results = checkpoint_dict["previous_epoch_results"]


def status_func(status_number, epoch_number, epoch_results):
    checkpoint_status_func(save_path, checkpoint_dict, epoch_results)

epoch_results = iterate_function(fit_function, [X, y], minibatch_size,
                                 list_of_output_names=["nll", "kl", "cost"],
                                 n_epochs=2000,
                                 status_func=status_func,
                                 previous_epoch_results=previous_epoch_results,
                                 shuffle=True,
                                 random_state=random_state)
