# AlgoriThemes

**Automated analysis of open-ended survey responses**

AlgoriThemes is a Streamlit application that helps researchers and
organisations make sense of written survey feedback. It identifies
recurring topics, gives them understandable descriptions, and---when
selected---compares positive, neutral and negative responses within
those topics. Users can explore the findings and export coded data and
reports.

The application is designed for people who work with survey responses,
including researchers, monitoring and evaluation professionals, HR teams
and other feedback analysts. No programming knowledge is needed to use
the interface.

> **Project status:** Research/capstone application under active
> development. Automated findings are intended to support, not replace,
> human interpretation.

## Features

-   **Two analysis routes:** topic discovery only, or topic discovery
    followed by sentiment analysis.
-   **CSV and Excel input:** select a worksheet (for Excel), an
    open-ended response column, and an optional respondent identifier.
-   **Data preparation:** remove blank responses; optionally handle
    duplicate text or duplicate respondent IDs; exclude responses with
    fewer than three words after cleaning.
-   **Topic discovery:** group semantically related responses and
    identify recurring subjects.
-   **Readable topic labels:** generate topic names and descriptions
    from the discovered topics.
-   **Broader themes when useful:** group related topics into themes
    when nine or more topics are found; otherwise show topics directly.
-   **Sentiment analysis (optional):** classify each retained response
    as positive, neutral or negative and compare the distribution within
    topics.
-   **Results exploration:** view charts, topic descriptions,
    representative responses, and response-level codes.
-   **Exports:** download coded responses and a coding dictionary
    (CSV/Excel), a Word report, and a standalone interactive HTML
    report.
-   **Real-data demos:** load configured research datasets and run the
    same analysis workflow used for uploaded data.

## How it works

``` text
Upload CSV/Excel or choose a demo
              |
Choose response column and prepare data
              |
      Discover topics
              |
  Name and describe topics
              |
Group into broader themes (9+ topics)
              |
  Optional sentiment analysis
              |
Explore, review and export results
```

The sentiment route discovers topics first, then assigns sentiment
labels to responses so users can compare opinions *within* a topic.
Sentiment describes the overall response; it is not aspect-level
sentiment for every subject mentioned in a multi-topic response.

### Models and technologies

  -----------------------------------------------------------------------------------------------------------------------------------
  Component               Technology                                                                          Role
  ----------------------- ----------------------------------------------------------------------------------- -----------------------
  Interface               [Streamlit](https://streamlit.io/)                                                  Upload, configuration,
                                                                                                              analysis and results

  Topic discovery         [BERTopic](https://maartengr.github.io/BERTopic/)                                   Unsupervised discovery
                                                                                                              of recurring topics

  Text embeddings         [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)                Represent responses by
                                                                                                              semantic similarity

  Sentiment               [CardiffNLP                                                                         Positive, neutral and
  classification          RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)   negative labels

  Topic interpretation    [Google Gemini](https://ai.google.dev/gemini-api/docs)                              Topic names,
                                                                                                              descriptions, broader
                                                                                                              themes and
                                                                                                              recommendations

  Data and charts         pandas, Plotly                                                                      Preparation,
                                                                                                              exploration and
                                                                                                              visualisation
  -----------------------------------------------------------------------------------------------------------------------------------

Sentiment predictions below **70% confidence** are flagged for review.
This threshold is a review aid, not a guarantee of correctness. Topic
names, descriptions and recommendations are AI-generated and should be
checked against the original responses.

## Project structure

The repository intentionally keeps the current flat Python-module
structure:

``` text
AlgoriThemes/
├── app.py                 # Streamlit interface and navigation
├── data_processing.py     # Text cleaning and preprocessing
├── demo_data.py           # Real-data demo paths and settings
├── topic_analysis.py      # Topic modelling workflow
├── sentiment_analysis.py  # Topic-level sentiment workflow
├── gemini_analysis.py     # AI-assisted topic interpretation
├── recommendations.py     # Recommendation generation
├── report_builder.py      # Word report generation
├── html_export.py         # Standalone interactive HTML export
├── outputs.py             # Export and summary helpers
├── demo_datasets/         # Real datasets (see data-sharing note)
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment variable names
├── .gitignore             # Files excluded from Git
└── README.md
```

A logo and screenshots may be added later; neither is required to run
the application.

## Run locally

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR_USERNAME/AlgoriThemes.git
cd AlgoriThemes
```

Replace `YOUR_USERNAME` with the GitHub account that owns the
repository. If you have not created the GitHub repository yet, you can
instead run the commands below in your existing project directory.

### 2. Create a Python environment

Python **3.11** is a reasonable starting point for this dependency
stack. If you already have a working `survey-studio` Conda environment,
you can reuse it.

``` bash
conda create -n survey-studio python=3.11 -y
conda activate survey-studio
pip install -r requirements.txt
```

Installing BERTopic, PyTorch and sentence-transformer dependencies may
take a while. On first use, the application may also download model
files.

### 3. Configure Gemini API credentials

Create a local `.env` file in the project root (do **not** commit it):

``` dotenv
GEMINI_API_KEY_1=your_api_key_here
```

The current Gemini integration looks for `GEMINI_API_KEY_1` through
`GEMINI_API_KEY_4`; one valid key is sufficient. Additional keys are
optional. Follow `.env.example` for any other settings used by your
version of the app.

**Never commit API keys, `.env`, or Streamlit secrets to GitHub.**

### 4. Start the application

``` bash
streamlit run app.py
```

Open the local address printed by Streamlit (usually
`http://localhost:8501`). Choose an analysis route, upload a dataset,
select the response column, prepare the responses, and run the analysis.

## Real-data demos

The homepage includes two demos, both of which use **real datasets**,
not fabricated examples or precomputed results.

  --------------------------------------------------------------------------
  Demo                    Dataset                 Configured response column
  ----------------------- ----------------------- --------------------------
  Explore topics          Faculty Support         `FacultySupportResponse`
                          Responses               

  Explore topics and      Online Employee Reviews `Text`
  sentiment                                       
  --------------------------------------------------------------------------

`demo_data.py` stores each dataset's relative path, worksheet, response
column, optional respondent ID, and survey question. By default, the
demo loader expects files similar to:

``` text
demo_datasets/
├── faculty_support.xlsx
└── online employee review dataset.xlsx
```

If your filenames, sheet names or column names differ, update
`DEMO_CONFIG` in `demo_data.py`. The demo buttons require those files to
exist **on the machine or server running Streamlit**.

**Data-sharing and deployment:** Before publishing demo datasets in a
public GitHub repository, confirm that their licenses allow
redistribution and that they contain no identifying or confidential
information. If you keep them out of Git, the deployed demos will not
work until you arrange a permitted way to make the files available on
the server. The main upload-and-analyse workflow can still work without
the demos.

## Publish the code to GitHub

Create an empty GitHub repository (for example, `AlgoriThemes`) without
adding a second README. From your existing project folder, run:

``` bash
git init
git add .
git status
git commit -m "Initial AlgoriThemes application"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AlgoriThemes.git
git push -u origin main
```

Before committing, review `git status` carefully to ensure `.env`,
private datasets, caches and other sensitive files are not staged. If
the folder is already a Git repository, do not repeat `git init` or add
a duplicate remote; check `git remote -v` first. Replace `YOUR_USERNAME`
with your GitHub username.

## Deploy with Streamlit Community Cloud

1.  Push the application code and `requirements.txt` to a GitHub
    repository.

2.  Visit [Streamlit Community Cloud](https://share.streamlit.io/) and
    create a new app from that repository.

3.  Select the repository, branch, and **`app.py`** as the main file.

4.  In the deployment's **Secrets** settings, add your Gemini key:

    ``` toml
    GEMINI_API_KEY_1 = "your_api_key_here"
    ```

5.  Deploy the app and test the upload route, both analysis routes,
    report exports, and (if available on the server) the demos.

The Gemini integration reads the API key from local environment
variables or Streamlit secrets. Do not upload your local `.env` file.

**Hosting considerations:** BERTopic, sentence-transformer embeddings
and RoBERTa can require significant memory and take time to download or
initialise. Resource limits on hosted platforms may affect performance
or prevent larger analyses. Test deployment with a small permitted
dataset before sharing the public link. Hosting and model inference are
not yet verified for this repository.

## Outputs

Depending on the analysis route and available results, AlgoriThemes can
provide:

-   Topic and, when applicable, broader-theme summaries and
    visualisations.
-   Sentiment distributions and within-topic comparisons.
-   Representative original responses and a searchable/filterable
    Response Explorer.
-   Response-level topic/theme codes and a coding dictionary.
-   Downloadable CSV and Excel data, Word reports, and interactive HTML
    reports.
-   AI-assisted recommendations grounded in the analysed responses.

## Limitations and responsible use

-   Topic modelling is unsupervised: topics may overlap, be too broad,
    or leave some responses unassigned.
-   Sentiment is a model prediction and can misread sarcasm, context,
    domain-specific language or mixed opinions.
-   Sentiment labels reflect each response overall, not every individual
    aspect mentioned in it.
-   AI-generated topic descriptions and recommendations need human
    review.
-   Sampled responses may be sent to an external AI provider for topic
    interpretation and recommendations. Remove personal identifiers and
    follow applicable data-handling requirements before analysing
    sensitive data.
-   Results depend on the dataset, preprocessing choices and model
    behaviour; they should not be treated as validated ground truth.

## Before publishing the repository

Check that `.gitignore` excludes at least:

``` gitignore
.env
.env.*
!.env.example
__pycache__/
*.py[cod]
.venv/
venv/
.streamlit/secrets.toml
```

Decide separately whether `demo_datasets/` may be published. If not, add
`demo_datasets/` to `.gitignore` and provide a permitted deployment
alternative for the demos.

To check what Git will commit:

``` bash
git status
git check-ignore -v .env
```

If `.env` has **ever** been committed, ignoring it now does not remove
it from history; rotate exposed keys and clean the repository history as
appropriate.

## Project background

AlgoriThemes was developed as a data science capstone project focused on
automating the analysis of open-ended survey responses. It brings
together unsupervised topic modelling, sentiment classification and
AI-assisted interpretation in an accessible research workflow.

------------------------------------------------------------------------

*AlgoriThemes supports exploratory analysis and research interpretation.
Review automated findings before using them to inform consequential
decisions.*
