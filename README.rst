Slides will be uploaded to https://speakerdeck.com/kastnerkyle

PDF version of slides included

To run the code in the graphics directory, you will need Theano.
I have not run this on CPU yet, but it runs pretty quickly on GPU.
To run the code, simply go to ``graphics_code/vae`` or ``graphics_code/cvae``

``THEANO_FLAGS="floatX=float32,device=gpu,mode=FAST_RUN" python vae.py``

or

``THEANO_FLAGS="floatX=float32,device=gpu,mode=FAST_RUN" python cvae.py``

will start training the model. After training,

``THEANO_FLAGS="floatX=float32,device=gpu,mode=FAST_RUN" python flying_vae.py serialized_vae.pkl``

or

``THEANO_FLAGS="floatX=float32,device=gpu,mode=FAST_RUN" python flying_cvae.py serialized_cvae.pkl``

will generate plots for the saved model.

Linked content
==============
sklearn-theano, a scikit-learn compatible library for using pretrained networks http://sklearn-theano.github.io/
My research code https://github.com/kastnerkyle/santa_barbaria
Neural network tutorial by @NewMu / Alec Radford https://github.com/Newmu/Theano-Tutorials
Theano Deep Learning Tutorials http://deeplearning.net/tutorial/
