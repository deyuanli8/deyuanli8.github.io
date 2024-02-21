let pyodideInstance = null;

window.onload = () => {
    // Load Pyodide and required packages
    async function loadPyodideAndPackages() {
        pyodideInstance = await loadPyodide({
            indexURL : "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/"
        });
        await pyodideInstance.loadPackage(['numpy', 'pandas', 'scipy', 'scikit-learn', 'micropip']);
        await pyodideInstance.runPythonAsync(`
            import micropip
            await micropip.install('openpyxl')
        `);
        return pyodideInstance;
    }

    let pyodideReady = loadPyodideAndPackages();
    let isProcessing = false;
    document.getElementById('submitBtn').addEventListener('click', async (event) => {
        if (isProcessing) {
            console.log("Processing is already in progress.");
            return; // Early exit if a process is already running
        }
        isProcessing = true;
        event.preventDefault(); // Prevent the default form submission
        let fileInput = document.getElementById('fileInput');
        let runtimeSelect = document.getElementById('runtimeSelect');
        let columnNamesInput = document.getElementById('columnNamesInput');
        let file = fileInput.files[0];
        let runtime = runtimeSelect.value;
        let categoricalColumns = columnNamesInput.value ? columnNamesInput.value.split(',') : [];

        if (file && runtime) {
            disableInputs();
            document.getElementById('downloadLink').style.display = 'none';
            document.getElementById('processingMessage').style.display = 'block'; // Show processing message
            // resetInputs()
            
            if (file.name.endsWith('.csv') || file.name.endsWith('.xlsx')) {
                let reader = new FileReader();
                reader.onload = async (e) => {
                    let arrayBuffer = e.target.result;
                    let data = new Uint8Array(arrayBuffer);

                    await pyodideReady;
                    const fileName = file.name;
                    pyodideInstance.FS.writeFile(fileName, data);

                    // Prepare and run Python code using Pyodide
                    let pythonCode = `
import numpy as np
import pandas as pd
from scipy.linalg import qr, qr_update
from sklearn.linear_model import LinearRegression

import time
print("HERE")

# minutes = 1
# categorical_columns = []
minutes = ${parseInt(runtime)}  # Runtime from the select dropdown
categorical_columns = ${JSON.stringify(categoricalColumns)}  # Categorical columns from the textarea input

# Use minutes and categorical_columns in your Python code as needed
print(f"Runtime: {minutes} minutes")
print(f"Categorical Columns: {categorical_columns}")

try:
    if '${fileName}'.endswith('.csv'):
        df = pd.read_csv('${fileName}')
    else:
        # Attempt to use openpyxl to read Excel file
        df = pd.read_excel('${fileName}', engine='openpyxl')
except Exception as e:
    raise Exception("Unable to process file. Please ensure it is a CSV or a supported Excel format.")
print(df)


def compute_discrepancy(df, disp = False):
    characteristics = [col for col in df.columns if 'Characteristic' in col]
    m = len(characteristics)

    cols_to_multiply = [f'Characteristic_{i}' for i in range(m)]
    multiplied_df = df[cols_to_multiply].multiply(df['x'], axis=0)
    multiplied_df.columns = [f'SignedChar_{i}' for i in range(m)]
    df.drop(columns=[col for col in df.columns if 'SignedChar' in col], errors='ignore', inplace = True)
    df = pd.concat([df, multiplied_df], axis=1)
    columns = [col for col in df.columns if 'SignedChar' in col]
    if disp:
        print(df[columns].sum())
    return df[columns].sum().abs().max()

def carole(df_vecs, c = None, p = 3, num_iter = 3, cols = None, initialize = True, iter_all = True):
    def potential(c, p, m, ser):
        curSum = 0
        for i in range(m):
            s = ser.iloc[i]
            if c*m < s*s:
                curSum+= 1e9
            else:
                curSum+= pow(c, p) * pow(m, p-1) / pow(c*m-s*s, p)
        return curSum
    if(len(df_vecs) == 0):
        return df_vecs
    df = df_vecs.copy()
    m = len([col for col in df.columns if 'Characteristic' in col])
    if cols is None:
        cols = df.index
    if initialize:
        df.loc[cols, 'x'] = 2*np.random.binomial(1, 0.5, len(cols)) - 1

    if c is None:
        s = compute_discrepancy(df)
        c = 1.1 * s*s/m

    iter_cols = df.index if iter_all else cols

    cols_to_multiply = [f'Characteristic_{i}' for i in range(m)]
    multiplied_df = df[cols_to_multiply].multiply(df['x'], axis=0)
    multiplied_df.columns = [f'SignedChar_{i}' for i in range(m)]
    df.drop(columns=[col for col in df.columns if 'SignedChar' in col], errors='ignore', inplace = True)
    df = pd.concat([df, multiplied_df], axis=1)

    columns = [col for col in df.columns if 'SignedChar' in col]
    st = df[columns].sum()
    for i in range(num_iter):
        isMin = True
        for j in iter_cols:
            if potential(c, p, m, st) > potential (c, p, m, st-2*df.loc[j][columns]):
                st-=2*df.loc[j][columns]
                df.loc[j, columns + ['x']] *=-1
                isMin = False
        if isMin:
            break
    return df

def local_search(df_vecs, p = 4, num_iter = 10, cols = None, initialize = True, iter_all = True):
    def lp(ser, p):
        return pow(ser.abs(), p).sum()
    if(len(df_vecs) == 0):
        return df_vecs
    df = df_vecs.copy()
    characteristics = [col for col in df.columns if 'Characteristic' in col]
    m = len(characteristics)

    if cols is None:
        cols = df.index
    if initialize:
        g = np.random.multivariate_normal(np.zeros(m), np.eye(m),1)[0]
        model = LinearRegression(fit_intercept = False)
        reg = model.fit(df.loc[cols][characteristics].T, g)
        df.loc[cols, 'x'] = np.sign(reg.coef_).astype(int)

    iter_cols = df.index if iter_all else cols

    cols_to_multiply = [f'Characteristic_{i}' for i in range(m)]
    multiplied_df = df[cols_to_multiply].multiply(df['x'], axis=0)
    multiplied_df.columns = [f'SignedChar_{i}' for i in range(m)]
    df.drop(columns=[col for col in df.columns if 'SignedChar' in col], errors='ignore', inplace = True)
    df = pd.concat([df, multiplied_df], axis=1)

    columns = [col for col in df.columns if 'SignedChar' in col]
    st = df[columns].sum()
    for i in range(num_iter):
        isMin = True
        for j in iter_cols:
            if lp(st, p) > lp(st-2*df.loc[j][columns], p):
                st-=2*df.loc[j][columns]
                df.loc[j, columns + ['x']] *=-1
                isMin = False
        if isMin:
            break
    return df

def soft(df_vecs, t = 1, num_iter = 3, cols = None, initialize = True, iter_all = True):
    def f(t, ser):
        return 1/t * np.log(np.exp(t*ser.abs()).sum())
    if(len(df_vecs) == 0):
        return df_vecs
    df = df_vecs.copy()

    m = len([col for col in df.columns if 'Characteristic' in col])
    if cols is None:
        cols = df.index
    if initialize:
        df.loc[cols, 'x'] = 2*np.random.binomial(1, 0.5, len(cols)) - 1
    iter_cols = df.index if iter_all else cols

    cols_to_multiply = [f'Characteristic_{i}' for i in range(m)]
    multiplied_df = df[cols_to_multiply].multiply(df['x'], axis=0)
    multiplied_df.columns = [f'SignedChar_{i}' for i in range(m)]
    df.drop(columns=[col for col in df.columns if 'SignedChar' in col], errors='ignore', inplace = True)
    df = pd.concat([df, multiplied_df], axis=1)

    columns = [col for col in df.columns if 'SignedChar' in col]
    st = df[columns].sum()
    for i in range(num_iter):
        isMin = True
        for j in iter_cols:
            if f(t, st) > f (t, st-2*df.loc[j][columns]):
                st-=2*df.loc[j][columns]
                df.loc[j, columns + ['x']] *=-1
                isMin = False
        if isMin:
            break
    return df

def faster_reduction(df_vecs):
    df = df_vecs.copy().reset_index(drop = True)
    m = len(df.columns)
    n = len(df)
    df['x'] = 0.0
    if m>=n:
        return df
    columns = [col for col in df.columns if 'Characteristic' in col]

    q, r = qr(df.iloc[:m+1][columns])
    col_order = list(range(m+1))
    last_ind = m
    for i in range(n - m):
        df['lambda'] = 0.0
        df.loc[col_order, 'lambda'] = q[:,-1]

        alpha_pos = (1 - df[(abs(df['x'])!=1) & (df['lambda']!=0)]['x'])/df[(abs(df['x'])!=1) & (df['lambda']!=0)]['lambda']
        alpha_neg = (-1 - df[(abs(df['x'])!=1) & (df['lambda']!=0)]['x'])/df[(abs(df['x'])!=1) & (df['lambda']!=0)]['lambda']
        if len(alpha_pos) == 0:
            idx = alpha_neg.abs().idxmin()
            alpha = alpha_neg.loc[idx]
        elif len(alpha_neg) == 0:
            idx = alpha_pos.abs().idxmin()
            alpha = alpha_pos.loc[idx]
        else:
            idx_pos = alpha_pos.abs().idxmin()
            idx_neg = alpha_neg.abs().idxmin()
            (_, alpha, idx) = min((abs(alpha_pos.loc[idx_pos]), alpha_pos.loc[idx_pos], idx_pos), (abs(alpha_neg.loc[idx_neg]), alpha_neg.loc[idx_neg], idx_neg))
        df['x'] = df['x']+alpha*df['lambda']
        df.loc[idx, 'x'] = np.sign(df.loc[idx, 'x'])

        u = np.zeros(m+1)
        col_order_idx = col_order.index(idx)
        u[col_order_idx] = -1
        last_ind+=1
        if last_ind < n:
            qr_update(q, r, u, df.loc[idx][columns] - df.loc[last_ind][columns], True)
            col_order[col_order_idx] = last_ind
    df.drop(columns = 'lambda', inplace = True)
    return df

def solve_reduced_local(df_vecs, t = 10):
    df = df_vecs.copy()
    df_remaining = df[abs(df['x'])!=1].copy()
    df_remaining['x'] = (df_remaining['x']*pow(2,t)).round(0).astype(int)
    for i in range(t):
        df_remaining.loc[df_remaining['x']%2 == 1, 'x'] += local_search(df_remaining[df_remaining['x'] % 2 == 1], 4)['x']
        df_remaining['x']/=2
    df_remaining.loc[df_remaining['x'] == 0, 'x'] = local_search(df_remaining[df_remaining['x'] == 0], 4)['x']
    df.loc[abs(df['x'])!=1, 'x'] = df_remaining['x']
    return df

def solve_reduced_carole(df_vecs, t = 10):
    df = df_vecs.copy()
    df_remaining = df[abs(df['x'])!=1].copy()
    df_remaining['x'] = (df_remaining['x']*pow(2,t)).round(0).astype(int)
    for i in range(t):
        df_remaining.loc[df_remaining['x']%2 == 1, 'x'] += carole(df_remaining[df_remaining['x'] % 2 == 1])['x']
        df_remaining['x']/=2
    df_remaining.loc[df_remaining['x'] == 0, 'x'] = carole(df_remaining[df_remaining['x'] == 0])['x']
    df.loc[abs(df['x'])!=1, 'x'] = df_remaining['x']
    return df

def solve_reduced_soft(df_vecs, t = 10):
    df = df_vecs.copy()
    df_remaining = df[abs(df['x'])!=1].copy()
    df_remaining['x'] = (df_remaining['x']*pow(2,t)).round(0).astype(int)
    for i in range(t):
        df_remaining.loc[df_remaining['x']%2 == 1, 'x'] += soft(df_remaining[df_remaining['x'] % 2 == 1])['x']
        df_remaining['x']/=2
    df_remaining.loc[df_remaining['x'] == 0, 'x'] = soft(df_remaining[df_remaining['x'] == 0])['x']
    df.loc[abs(df['x'])!=1, 'x'] = df_remaining['x']
    return df

def normalize_data(df_vecs):
    return df_vecs.apply(lambda col: np.ones(len(col)) if col.nunique() == 1 and col.iloc[0] != 0 else np.zeros(len(col)) if col.nunique() == 1 else (2*(col - col.min())/(col.max() - col.min()) - 1), axis=0)

def get_split(df_vecs, minutes = 5, categorical_columns = None):
    def find_best(df_current, current_discrepancy, df_new):
        new_discrepancy = compute_discrepancy(df_new, disp = False)
        if current_discrepancy <= new_discrepancy:
            return (df_current, current_discrepancy)
        print("Better Discrepancy Found:", new_discrepancy)
        return (df_new, new_discrepancy)
    start_time = time.time()
    df = df_vecs.copy()
    if categorical_columns is not None:
        if not set(categorical_columns).issubset(set(df.columns)):
            print("Error: Inputted list of categorical columns are not part of dataframe.")
            raise Exception("Please ensure all inputted categorical column are column names in the uploaded Excel or CSV file.")
    df_encoded = pd.get_dummies(df, columns=categorical_columns).astype(float)
    df_encoded.columns = [f'Characteristic_{i}' for i in range(len(df_encoded.columns))]
    m = len(df_encoded.columns)
    df_normalized = normalize_data(df_encoded)
    df_reduced = faster_reduction(df_normalized)


    df_best = local_search(df_reduced, 4, cols = df_reduced[abs(df_reduced['x'])!=1].index)
    best_discrepancy = compute_discrepancy(df_best, disp = False)
    print("Starting Discrepancy:", best_discrepancy)
    df_best, best_discrepancy = find_best(df_best, best_discrepancy, soft(df_reduced, 1, cols = df_reduced[abs(df_reduced['x'])!=1].index))
    df_best, best_discrepancy = find_best(df_best, best_discrepancy, carole(df_reduced, cols = df_reduced[abs(df_reduced['x'])!=1].index))

    df_best, best_discrepancy = find_best(df_best, best_discrepancy, solve_reduced_local(df_reduced))
    df_best, best_discrepancy = find_best(df_best, best_discrepancy, solve_reduced_soft(df_reduced))
    df_best, best_discrepancy = find_best(df_best, best_discrepancy, solve_reduced_carole(df_reduced))

    soft_const = 2
    local_const = 6
    max_soft = int(np.log(1e307 / m)/(best_discrepancy+2))
    max_local = int(np.log(1e307 / m)/np.log(best_discrepancy+2))
    while time.time() - start_time < minutes*60:
        for method in ['Local', 'Soft', 'Carole']:
            if method == 'Local' and local_const < max_local:
                df_new = local_search(df_best, local_const, initialize = False)
                local_const+=1
            elif method == 'Soft' and soft_const < max_soft:
                df_new = soft(df_best, soft_const, initialize = False)
                soft_const +=1
            else:
                df_new = carole(df_best, initialize = False)
            df_best, best_discrepancy = find_best(df_best, best_discrepancy, df_new)
            max_soft = int(np.log(1e307 / m)/(best_discrepancy+2))
            max_local = int(np.log(1e307 / m)/np.log(best_discrepancy+2))
    print("Final Discrepancy:", best_discrepancy)
    df['discrepancy split'] = df_best['x']
    return df

df_output = get_split(df, minutes, categorical_columns)


output = df_output.to_csv(index=False)
output
                    `;
                    let output = await pyodideInstance.runPython(pythonCode);

                    // Handle the output
                    document.getElementById('processingMessage').style.display = 'none'; // Hide the processing message
                    resetInputs();
                    createDownloadLink(output, fileName);
                    enableInputs();
                };
                // Read the file as an ArrayBuffer for both CSV and Excel files
                reader.readAsArrayBuffer(file);
            } else {
                alert('Please upload a CSV or Excel file.');
                document.getElementById('processingMessage').style.display = 'none'; // Hide processing message if file is invalid
                resetInputs(); // Reset inputs even if file is not valid
                enableInputs();
            }
        } else {
            alert('Please select a file and runtime.');
            document.getElementById('processingMessage').style.display = 'none'; // Hide processing message if file is invalid
            resetInputs(); // Reset inputs even if conditions are not met
            enableInputs();
        }
        isProcessing = false;
    });

    // Function to create a downloadable link for the processed CSV data
    function createDownloadLink(csvData, processed_fileName) {
        const blob = new Blob([csvData], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const link = document.getElementById('downloadLink');
        link.href = url;
        link.download = 'processed_'+processed_fileName;
        link.style.display = 'block';
    }

    function resetInputs() {
        document.getElementById('fileInput').value = '';
        document.getElementById('runtimeSelect').value = '';
        document.getElementById('columnNamesInput').value = '';
        // Additional UI adjustments if necessary
    }

    function disableInputs() {
        document.getElementById('submitBtn').disabled = true;
        document.getElementById('fileInput').disabled = true;
        document.getElementById('runtimeSelect').disabled = true;
        document.getElementById('columnNamesInput').disabled = true;
    }

    function enableInputs() {
        // List of elements to re-enable
        const elements = [
            document.getElementById('fileInput'),
            document.getElementById('runtimeSelect'),
            document.getElementById('columnNamesInput'),
            document.getElementById('submitBtn')
        ];

        elements.forEach(element => {
            setTimeout(() => {
                element.disabled = false;
            }, 100); // Delay re-enabling to ensure queued events are not executed
        });
    }
};
