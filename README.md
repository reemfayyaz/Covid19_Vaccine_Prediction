# COVID-19 Vaccine Acceptance Predictor

A simple Flask web app that predicts whether a person is likely to **accept**
or be **hesitant** about a COVID-19 vaccine, using a pre-trained
`RandomForestClassifier` (scikit-learn).

## How it works

The user fills out a short form (age, gender, state, health history, trust
level, etc.), the app encodes the inputs to match the model's training
format, and returns a prediction with a confidence score.

## Project structure

```
covid_app/
├── app.py                 # Flask app (routes + prediction logic)
├── requirements.txt       # Python dependencies
├── Covid_19_model.pkl     # Pre-trained RandomForestClassifier
└── templates/
    └── index.html         # Form + results page
```

## Model details

- **Type:** `sklearn.ensemble.RandomForestClassifier` (100 estimators)
- **Output:** binary — `0` (hesitant) or `1` (likely to accept)
- **Input features (in order):**
  `Age`, `Gender`, `State`, `Chronic_Disease`, `Previous_COVID`,
  `Trust_in_Vaccine`, `Fear_of_Side_Effects`, `Vaccine_Type_Preference`

## ⚠️ Before you run this

The model was trained on **numerically encoded** categorical fields, but the
exact encoding used at training time isn't stored in the pickle file. `app.py`
contains placeholder mappings (`GENDER_MAP`, `STATE_MAP`, `YES_NO_MAP`,
`TRUST_FEAR_MAP`, `VACCINE_TYPE_MAP`) near the top of the file — **you must
update these to match whatever encoding you used when training the model**,
or predictions will be inaccurate.

## Setup

1. Clone the repo and move into it:
   ```bash
   git clone <your-repo-url>
   cd covid_app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   python app.py
   ```

5. Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Note on the model file size

`Covid_19_model.pkl` is roughly 216 MB, which exceeds GitHub's 100 MB limit
for regular file pushes. To include it in this repo, use
[Git LFS](https://git-lfs.github.com/):

```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes Covid_19_model.pkl
git commit -m "Track model with Git LFS"
```

Alternatively, host the model file elsewhere (e.g. Hugging Face Hub, S3,
Google Drive) and download it at app startup instead of committing it to the
repo.

## Disclaimer

This tool is for educational/demonstration purposes only and should not be
used to make real public health or medical decisions.
