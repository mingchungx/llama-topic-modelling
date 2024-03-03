# llama-topic-modelling

Analyze the data using topic modelling machine learning to group various topics in the LocalLLaMa subreddit, [LocalLLaMa Subreddit](https://www.reddit.com/r/LocalLLaMA/new/).

Our research approach uses topic modelling with Latent Dirichlet Allocation (LDA) algorithm for unsupervised machine learning pipelining. The analyzer collects data from the LocalLLaMA subreddit, then runs topic clustering on queries related-to, and not related-to privacy. The data is read fetched from the Reddit API and tokened to multi-dimensional vectors. Then, dimensional reduction is applied to visualize and analyze demand for privacy and security in LocalLLaMA.

## Usage/Examples

Generate output `lda_visualization.html` file by running the following commands.

````bash

```bash
chmod +x run.sh
./run.sh
````

Visualize the data opening the `output/lda_visualization.html` file in a web browser.
