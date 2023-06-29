Drive link : https://drive.google.com/drive/u/1/folders/1FdgHtD_MbqfDB-pH6U_6SU39_ZRLvbTq

# Evaluation Metric for Personalized Summarization
Developing an evaluation metric "Effective degree of insensitivity (e-DINS)" for better ranking personalized text summarization models.
## e-DINS<sub>sub</sub>
## e-DINS<sub>TV</sub>
The study aims to introduce a new metric e − DINS<sub>TV</sub> which considers the rate of change of model generated summaries when compared to rate of change of user profiles with respect to time. The perception of saliency is subjective and in case of temporal variance, it changes overtime for each user.
## {PENS}: A Dataset and Generic Framework for Personalized News Headline Generation
This is a Pytorch implementation of [PENS](https://www.microsoft.com/en-us/research/uploads/prod/2021/06/ACL2021_PENS_Camera_Ready_1862_Paper.pdf). 

## I. Guidance

### 0. Enviroment
- Install pytorch version >= '1.4.0'
- Install the pensmodule package under ''PENS-Personalized-News-Headline-Generation'' using code ``` pip install -e . ```

### 1. Data Prepare
- Download the PENS dataset [here](https://msnews.github.io/pens.html) and put the dataset under data/.
- (optional) Download glove.840B.300d.txt under data/ if you choose to use pretrained glove word embeddings.

### 2. Running Code
- ```cd pensmodule ```
- Follow the order: Preprocess --> UserEncoder --> Generator and run the pipeline**.ipynb notebook to preprocess, train the user encoder and the train generator, individually.
- The model generated summaries will be saved with the name "hyps_{model_name}" in the same directory.

More infor please refer to the homepage of the [introduction of PENS dataset](https://msnews.github.io/pens.html).


## II. Data Manipulation for Evaluation
- Collect the model generated summaries and move it to "temporal_variance" directory.
- Run the dataset generation notebook to prepare the dataset for evaluation.
- Place the news.tsv file in "temporal_variance" directory

## III. Calculation of e-DINS<sub>sub</sub>

## IV. Calculation of e-DINS<sub>TV</sub>
- Move the combined_dataset.pkl to "temporal_variance" directory to calculate the score
- Run the Temporal Variance Notebook according to the guided steps in the notebook
- 5-model-dataset.json file is also needed for calculation of rouge score
