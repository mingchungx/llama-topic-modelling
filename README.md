# llama-topic-modelling

Analyze the data using topic modelling machine learning to group various topics in the LocalLLaMa subreddit, [LocalLLaMa Subreddit](https://www.reddit.com/r/LocalLLaMA/new/).

Our research approach uses topic modelling with Latent Dirichlet Allocation (LDA) algorithm for unsupervised machine learning pipelining on r/LocalLLaMa. The pipeline collects over 1000 submission documents from the r/LocalLLaMA subreddit, then runs topic clustering on queries related-to the use cases and developer need for on-device and smaller LLaMA. The data is fetched from the Reddit API, sorted by relevance. Documents are tokened with multi-dimensional vector encoding, then dimensional reduction is applied to visualize and analyze demand for smaller, on-device applications in LocalLLaMA.

## Usage/Examples

Generate output `index.html` file by running the following commands.

````bash

```bash
chmod +x run.sh
./run.sh
````

Visualize the data opening the `output/index.html` file in a web browser.
