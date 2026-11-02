# DAMA61 – Written Assignment 1 (end-to-end ML projects)

Both problems follow the end-to-end checklist of Géron, *Hands-On Machine Learning*, Chapter 2:
big picture → get the data → discover & visualise → prepare → select & train → fine-tune → present → launch, monitor & maintain.

| File | Purpose |
|---|---|
| `problem1.py` | Bike Sharing – daily demand regression (assignment Q1–Q8 mapped onto the 8 steps; extras: current-solution baseline, attribute combinations, calendar events, time-based validation) |
| `problem2.py` | Breast Cancer – malignant/benign classification (Q1–Q6 mapped onto the 8 steps) |
| `app.py` | Streamlit app: one tab per step, live prediction form, model download, monitoring |
| `DAMA61_WA1_solution.ipynb` | Notebook answering the assignment questions in order |
| `setup_env.ps1` / `setup_env.sh` | Create the virtual environment (Windows / macOS-Linux) |
| `bike+sharing+dataset/day.csv` | Offline copy of the data (the script downloads it from UCI first) |

## 1. Create the isolated environment

A virtual environment keeps this project's libraries separate from the rest of the computer,
so versions cannot clash with other projects.

**Windows (PowerShell)** – from the WA1 folder:

```powershell
python -m venv .venv                      # create the environment (folder .venv)
.\.venv\Scripts\Activate.ps1              # activate it -> prompt shows (.venv)
python -m pip install --upgrade pip
pip install -r requirements.txt           # numpy, pandas, matplotlib, scipy, scikit-learn, joblib, streamlit
pip install jupyter ipykernel             # optional, for the notebook
python -m ipykernel install --user --name dama61-wa1
```

If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
Or simply run `.\setup_env.ps1`, which does all of the above.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(or `source setup_env.sh`). Leave the environment with `deactivate`.

## 2. Run

```bash
python problem1.py          # downloads + extracts the data, prints each step, figures -> ./figures, model -> ./models
python problem2.py          # add --no-show to only save the figures
streamlit run app.py        # interactive app at http://localhost:8501
jupyter notebook            # optional: open DAMA61_WA1_solution.ipynb, kernel "dama61-wa1"
```

### Running in Spyder (plots in the Plots pane)

1. In the activated environment: `pip install spyder-kernels` (if Spyder complains about the version,
   install the version it names, e.g. `pip install "spyder-kernels==3.0.*"`).
2. Spyder → **Tools > Preferences > Python interpreter** → *Use the following interpreter* →
   `WA1\.venv\Scripts\python.exe` → restart the console (**Consoles > Restart kernel**).
3. **Tools > Preferences > IPython console > Graphics** → Backend: **Inline**.
   In the **Plots** pane menu make sure *Mute inline plotting* is **off**.
4. Open `problem1.py` or `problem2.py` and press **F5** (Run). Spyder sets the working directory to the
   file's folder. Every figure appears in the Plots pane (and is still saved to `figures/`), and the
   printed results appear in the IPython console.

The scripts switch Spyder/Jupyter to inline plots themselves (`%matplotlib inline`) and display each
figure as soon as its step finishes. From a plain terminal the figures open in windows at the end
(`--no-show` = only save them).

## 3. Deploy on Streamlit Community Cloud

1. Push this folder to a **private** GitHub repository (`.gitignore` excludes `.venv`, downloads, figures, models).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Choose the repository, branch `main`, main file `app.py`, then **Deploy**.
