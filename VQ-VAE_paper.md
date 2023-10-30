# VQ-VAE

*VQ-VAE (vector-quantizing variational autoencoder) (https://arxiv.org/abs/1711.00937) is a VAE where instead of learning to map data through a Gaussian bottleneck, we use a categorical distribution as the bottleneck - that is, we compress data into discrete 1-of-K random variables. This alleviates some issues with the Gaussian VAE (poor reconstruction quality, posterior collapse), and lets us directly calculate how many bits of information can pass through the variational bottleneck.*

Combines VAE (variational autoencoder) framework with discrete latent space representations. 

**VAE**
- Encoder (recognition model): parameterises a posterior distribution $q(z|x)$ of discrete latent random variables $z$ given the input data $x$ - ie. maps the input data $x$ to a latent representation $z$ through $q(z|x)$
- Prior distribution: $p(z) \sim \mathcal{N}(0, I)$. Sampled from normal distribution where $I$ is the identy matrix - creating smoother latent space 
- Decoder (generative model): maps $z$ back to data space $x$ through the distribution $p(x|z)$

Posteriors and priors are assumed to to be normally distributed with diagonal covariance - allowing for Gaussian reparametrisation trick. 
- **Gaussian reparametrisation trick**: instead of sampling $z$ directly from $q(z|x)$ we sample an auxillary random variable $\epsilon \sim \mathcal{N}(\mu, \sigma^2)$ and then compute:
$$
z = \mu(x) +\Sigma(x)^{0.5}\odot\epsilon
$$
*This allows the gradients to be backpropped through the sochastic nodes, enabling training of VAEs using gradient-methods*

- VAEs suffers from posterior collapse where the encoder maps different input data to similar or identical points in the latent space, causing a loss of information. This might lead to some latent variables being ignored

**VQ-VAE**

Introduces a categorical/discrete latent embedding space where 