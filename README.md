# AlgoriThemes

**From open-ended responses to interpretable topics, sentiment insights, and research-ready reports.**

AlgoriThemes is a research-focused **Streamlit application** for exploring qualitative survey data at scale. It combines unsupervised topic modelling, optional sentiment classification, and AI-assisted interpretation to help researchers identify recurring concerns, review the evidence behind each topic, and export structured findings.

The application is intended for researchers, monitoring and evaluation teams, human resources professionals, and other analysts working with open-ended feedback. The interface does not require programming experience.

> **Project status:** Data science capstone / active development. Outputs are exploratory and require human review; the application is **not a substitute for validated qualitative coding**.

## At a glance

| | |
|---|---|
| **Input** | CSV or Excel files containing open-ended responses |
| **Analysis modes** | Topic discovery; topic discovery + sentiment |
| **Topic model** | BERTopic with Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Sentiment model** | CardiffNLP RoBERTa (positive / neutral / negative) |
| **AI interpretation** | Google Gemini for topic labels, descriptions, themes, and recommendations |
| **Outputs** | Interactive exploration, response-level codes, CSV/Excel exports, Word reports, HTML reports |

## What the application does

- **Imports survey responses:** Upload CSV or Excel data, select the relevant Excel worksheet, choose a free-text response column, and optionally provide a respondent ID.
- **Prepares text for analysis:** Remove empty responses, optionally handle duplicate responses or respondent IDs, and apply the configured minimum cleaned-text length of three words.
- **Discovers topics:** Use semantic embeddings and BERTopic to group related responses and surface prominent topic keywords.
- **Explains the findings:** Generate accessible topic names and descriptions. When **nine or more topics** are found, the application can group related topics into broader themes.
- **Adds optional sentiment analysis:** Label retained responses as positive, neutral, or negative, then compare overall sentiment within discovered topics.
- **Supports evidence-based review:** Explore charts, representative original responses, topic descriptions, and response-level codes.
- **Exports reusable results:** Download coded data and a coding dictionary in CSV/Excel, plus Word and standalone interactive HTML reports.
- **Includes dataset-based demos:** Run configured faculty-support and employee-review datasets through the same application workflow, provided the files are available locally or on the host.

## How the analysis works

```text
CSV / Excel upload or configured demonstration dataset
                         |
           Select a free-text response column
                         |
           Clean and prepare the responses
                         |
         Generate sentence-level embeddings
                         |
            Discover topics with BERTopic
                         |
      Generate topic labels and descriptions (Gemini)
                         |
     Group related topics into broader themes, if applicable
                         |
       Optional response-level sentiment classification
                         |
             Explore, review, and export
```

### Technology stack

| Component | Technology | Purpose |
|---|---|---|
| Application interface | [Streamlit](https://streamlit.io/) | Interactive upload, settings, and results |
| Topic discovery | [BERTopic](https://maartengr.github.io/BERTopic/) | Discover recurring topics without predefined labels |
| Text representation | [Sentence Transformers](https://www.sbert.net/), `all-MiniLM-L6-v2` | Represent semantically similar responses with nearby vectors |
| Sentiment classification | [CardiffNLP RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | Predict positive, neutral, or negative sentiment |
| AI-assisted interpretation | [Google Gemini](https://ai.google.dev/gemini-api/docs) | Generate topic names, summaries, broader themes, and recommendations |
| Data processing and charts | pandas, Plotly | Manipulate responses and present visual findings |

**Topic discovery is not the same as sentiment analysis.** The model first identifies *what* respondents discuss. The optional sentiment step estimates *how positive, neutral, or negative* each entire response is. It does **not** attribute a separate sentiment to every subject mentioned within a multi-topic response.

Sentiment predictions with confidence **below 70%** are flagged for review. This is an operational review threshold, **not a calibrated probability of correctness**. All AI-generated labels and recommendations should be checked against source responses.

## Research context and model selection

The capstone's exploratory modelling work compared four approaches to identifying themes in student survey responses:

| Model | Core approach | Main distinction |
|---|---|---|
| **Latent Dirichlet Allocation** | Estimates probabilistic mixtures of topics and words | Word co-occurrence and topic probabilities |
| **Non-Negative Matrix Factorization** | Factorises a non-negative document–term matrix | Weighted word-pattern discovery |
| **K-Means Clustering** | Assigns vectorised responses to nearby centroids | Numerical similarity between documents |
| **BERTopic** | Uses semantic sentence embeddings, clustering, and topic representations | Groups ideas expressed with different vocabulary |

### Earlier notebook evaluation

An earlier **exploratory notebook**, using **1,603 retained faculty-support responses**, compared the four approaches at **3, 6, and 12 topics** using three measures:

| Metric | What it assesses | Highest-performing model/result in the notebook |
|---|---|---|
| **U_mass coherence** | Co-occurrence of representative topic words; higher (often less negative) is better
| **C_v coherence** | Association among representative topic words; higher is better
| **Mean intra-topic cosine similarity** | Semantic compactness of responses within assigned topics; higher is better 

BERTopic achieved the highest cosine similarity across the three tested configurations, which aligned with the project's objective of grouping differently worded responses that express similar ideas. **No model was best on every metric.**

**Important distinction:** Human-coded validation and independent testing remain outstanding.

## Getting started

### Prerequisites

- Python **3.11** (recommended starting point for this project; compatibility has not been exhaustively tested).
- Git, if cloning the repository.
- A valid **Google Gemini API key** for AI-assisted topic interpretation.
- Enough storage and memory to install and run transformer-based models. Model weights may download on first use.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AlgoriThemes.git
cd AlgoriThemes
```

Replace `YOUR_USERNAME` with the repository owner's GitHub handle. If the repository has not been published, use an existing local project folder instead.

### 2. Create an environment and install dependencies

Using Conda:

```bash
conda create -n survey-studio python=3.11 -y
conda activate survey-studio
python -m pip install -r requirements.txt
```

Alternatively, using Python's built-in virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

> Installing PyTorch, BERTopic, and embedding-model dependencies can be resource-intensive. Consult any installation errors before changing dependency versions.

### 3. Configure your Gemini key

Create a local `.env` file in the project root, using `.env.example` as the reference:

```dotenv
GEMINI_API_KEY_1=your_api_key_here
```

The current integration checks `GEMINI_API_KEY_1` through `GEMINI_API_KEY_4`. One valid key is sufficient; the others are optional. **Never commit real credentials** or a local `.env` file.

### 4. Start Streamlit

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit (typically `http://localhost:8501`). Choose an analysis route, upload your file, select the response column, review the preparation settings, and run the analysis.

### Input expectations

- A **CSV or Excel** file with at least one column containing open-ended written responses.
- For Excel, select the correct worksheet.
- An optional respondent ID column can help identify repeated records.
- Review whether removing duplicate text or very short responses is appropriate for your study before accepting those exclusions.

## Example datasets

The application describes two configured **real-data demonstrations**:

| Demo | Dataset | Response column |
|---|---|---|
| Discover topics | Faculty Support Responses | `FacultySupportResponse` |
| Discover topics + sentiment | Online Employee Reviews | `Text` |

The configuration is maintained in `demo_data.py` (`DEMO_CONFIG`). Expected filenames may resemble:

```text
demo_datasets/
├── faculty_support.xlsx
└── online employee review dataset.xlsx
```

These files **must be present on the computer or server running Streamlit**. Adjust the configured filename, worksheet, and column names if yours differ. The standard upload workflow does not require the demo files.

**Do not publish example data without confirming your right to redistribute it.** Remove direct and indirect identifiers, review applicable confidentiality obligations, and ensure that demonstration files are legally shareable.

## Outputs and interpretation

Depending on the selected route and available results, AlgoriThemes can produce:

1. **Topic summaries:** representative keywords, generated descriptions, and relevant response examples.
2. **Broader theme summaries:** higher-level categories when topic grouping is applicable.
3. **Sentiment breakdowns:** overall response sentiment and comparisons within topics.
4. **Response Explorer:** searchable or filterable original responses with assigned codes.
5. **Research exports:** coded response tables and coding dictionaries (CSV/Excel), Word reports, and interactive HTML reports.
6. **AI-assisted recommendations:** suggestions based on the analysed responses, subject to researcher review.

**Interpretation guidance:** A topic label is an AI-generated summary of a cluster, not a ground-truth code. Inspect representative *and atypical* responses, document any manual corrections, and avoid treating sentiment confidence as a guaranteed accuracy estimate.

## Repository structure

The project currently uses a flat Python-module layout:

```text
AlgoriThemes/
├── app.py                 # Streamlit interface and navigation
├── data_processing.py     # Cleaning and preprocessing
├── demo_data.py           # Demo file paths and settings
├── topic_analysis.py      # Topic modelling workflow
├── sentiment_analysis.py  # Sentiment workflow
├── gemini_analysis.py     # AI-assisted topic interpretation
├── recommendations.py     # Generated recommendations
├── report_builder.py      # Word report creation
├── html_export.py         # Interactive HTML export
├── outputs.py             # Data export and summary utilities
├── demo_datasets/         # Optional permitted datasets
├── requirements.txt       # Python dependencies
├── .env.example           # Credential variable examples
├── .gitignore             # Git exclusions
└── README.md
```

## Deploying the application

A possible deployment target is [Streamlit Community Cloud](https://share.streamlit.io/):

1. Push your application and `requirements.txt` to a GitHub repository.
2. Create a new Streamlit app using that repository and choose **`app.py`** as the entry point.
3. Add the Gemini credential to the deployed app's **Secrets** settings:

   ```toml
   GEMINI_API_KEY_1 = "your_api_key_here"
   ```

4. Deploy and verify file upload, both analysis modes, topic interpretation, report generation, and any server-hosted demos.

**Deployment is not yet verified for this repository.** Topic modelling and sentiment models can consume substantial memory and may exceed free-hosting resource limits. Test with a small, authorised dataset before using larger inputs. Never upload your local `.env` file.

### Publishing safely to GitHub

Before the first commit, confirm `.gitignore` excludes at least:

```gitignore
.env
.env.*
!.env.example
.streamlit/secrets.toml
__pycache__/
*.py[cod]
.venv/
venv/
```

Decide whether `demo_datasets/` is suitable for public release. If not, add that path to `.gitignore` and provide a permitted deployment alternative for the demos.

For a **new, not-yet-initialised** repository:

```bash
git init
git add .
git status
git commit -m "Initial AlgoriThemes application"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AlgoriThemes.git
git push -u origin main
```

Inspect `git status` before committing. If Git is already configured, check `git remote -v` instead of creating a duplicate remote. If credentials were previously committed, rotate them; adding `.env` to `.gitignore` does not remove existing Git history.

## Limitations and responsible use

| Limitation | Why it matters |
|---|---|
| **Dataset dependence** | Topics and model performance may change across domains, languages, sample sizes, or writing styles. |
| **Loss of short responses** | The default minimum cleaned-text length may exclude concise but informative feedback. |
| **Single-topic assignment and outliers** | Responses discussing several issues may be reduced to one dominant topic, and reassignment of outliers can introduce errors. |
| **AI-assisted interpretation** | Generated labels, broader themes, and recommendations can be incomplete, misleading, or sensitive to sampled examples. |
| **Limited external validation** | Coherence and similarity metrics do not replace independent, human-coded assessments of topic quality. |

Additional considerations:

- Sentiment predictions can misinterpret sarcasm, mixed opinions, and domain-specific language.
- Sentiment is assigned to the **response overall**, not independently to each aspect discussed.
- Selected response text may be sent to an **external AI provider** when AI interpretation or recommendations are requested. Confirm your organisation's data-sharing permissions and provider terms before using confidential or sensitive material.
- Uploaded-data retention, access controls, and production security have **not been independently audited** in this documentation.
- Do not use automated labels or recommendations as the sole basis for high-stakes individual decisions.

## Project direction

Priorities for further development and research validation include:

- Comparing automated topics with an independently human-coded sample.
- Assessing topic stability across repeated runs and different datasets.
- Reviewing the effect of excluding very short responses and reassigning outliers.
- Evaluating sentiment errors and whether confidence scores are calibrated.
- Testing memory requirements, upload limits, and deployment reliability.

---

**AlgoriThemes is a research-support tool.** Its value lies in making open-ended feedback easier to explore and organise while keeping interpretation, validation, and final decisions with researchers.

## Business opportunity: Who AlgoriThemes serves and how it improves their work

Organisations routinely collect open-ended feedback through surveys, evaluations, reviews, and consultations, but making sense of that feedback often requires hours of manual reading and coding. AlgoriThemes addresses this workflow by turning a spreadsheet of written responses into **reviewable topics, optional sentiment summaries, response-level codes, and exportable reports**. It is designed to **support analysts rather than replace their judgement**.

### Target users and practical value

| Target users | Typical challenge | How AlgoriThemes can improve their workflow |
|---|---|---|
| **Universities and education researchers** | Large volumes of course evaluations, student feedback, and open-ended research surveys | Surface recurring concerns about teaching and student support; compare themes across survey cycles; prepare coded datasets for further analysis. |
| **Non-governmental organisations and monitoring and evaluation teams** | Narrative feedback from programme participants, field surveys, and endline evaluations is difficult to synthesise consistently | Organise feedback into recurring themes, locate illustrative responses, and produce evidence summaries for programme reviews and donor reporting. |
| **Human resources and people-analytics teams** | Employee engagement surveys and workplace reviews contain dispersed, sometimes conflicting opinions | Identify common workplace concerns and, when enabled, inspect positive, neutral, and negative responses within each topic to support follow-up discussions. |
| **Customer experience and product teams** | Free-text customer feedback is scattered across surveys and product research | Group recurring product or service issues, examine representative comments, and turn qualitative feedback into a clearer starting point for prioritisation. |
| **Research agencies and consultancies** | Repetitive first-pass coding and reporting across multiple client studies | Standardise the initial exploratory workflow and export reusable coding tables, dictionaries, and client-ready report drafts for analyst review. |
| **Public-sector service and policy teams** | Open-ended citizen feedback and consultation submissions are time-consuming to review | Highlight recurring service-delivery concerns, make feedback easier to navigate, and support transparent human-led thematic review. |
| **Healthcare and service-quality researchers** | Patient-experience or service-feedback surveys can be lengthy and sensitive | Explore recurring experience themes and prepare qualitative summaries, **provided data are appropriately de-identified and sharing is authorised**. |

### The value proposition

- **Less repetitive preparation:** Bring data cleaning, topic discovery, interpretation, visual exploration, and export into one workflow instead of stitching together multiple tools.
- **More structured qualitative evidence:** Give analysts consistent starting-point topic assignments and a coding dictionary, while preserving access to the original responses for verification.
- **Better visibility into feedback:** Make recurring concerns and differences in overall sentiment easier to inspect rather than relying only on a handful of manually selected quotes.
- **Faster reporting cycles:** Use exportable tables and draft reports as inputs to researcher-reviewed findings, presentations, and action plans.
- **Accessible analysis:** Offer an interface for teams without dedicated machine learning or software development capacity.

These are **proposed workflow benefits**, not experimentally demonstrated time savings or validated improvements in decision quality. They should be tested in customer pilots against existing manual-analysis processes.

### Potential commercial approach

AlgoriThemes could be developed into a **business-to-business research and feedback analytics service**, with several possible delivery options:

1. **Self-service subscription:** Teams upload authorised survey files, run analyses, and export results within agreed usage limits.
2. **Project-based analysis:** Researchers, NGOs, or consultancies pay for a defined survey-analysis engagement with human-reviewed findings.
3. **Organisational deployment:** Institutions with stronger privacy requirements use a separately configured deployment, subject to security, access-control, and data-governance development.
4. **Research and implementation partnerships:** Universities and evaluation teams pilot the application on real projects and help validate topic quality, analyst time, and reporting usefulness.

These are **commercial opportunities under consideration**, not currently advertised plans or confirmed product capabilities.

### How success would be measured

A pilot should compare AlgoriThemes against an existing review process using **time to first coded draft**, **agreement with independent human coders**, **topic interpretability**, **analyst correction rates**, **report preparation effort**, and **user satisfaction**. For organisations handling sensitive information, access controls, retention policies, and external-model data sharing must be reviewed before deployment.

**Long-term opportunity:** Help organisations move from collecting feedback to systematically understanding and acting on it, while keeping the underlying evidence traceable and final interpretation under human control.

