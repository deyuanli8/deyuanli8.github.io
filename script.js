let pyodideInstance = null;
let initialRuntimeText = '';

document.addEventListener('DOMContentLoaded', function () {
    const runtimeSelect = document.getElementById('runtimeSelect');
    const runtimeToggle = document.getElementById('runtimeToggle');
    initialRuntimeText = runtimeToggle.textContent;
    for (let i = 0; i <= 10; i++) {
        const listItem = document.createElement('li');
        const option = document.createElement('a');
        option.classList.add('dropdown-item', 'negative-py-1');
        option.href = '#';
        option.textContent = `${i} min`;
        option.addEventListener('click', function (event) {
            event.preventDefault();
            runtimeToggle.textContent = `${i} min`;
            runtimeToggle.setAttribute('data-value', i);
        });
        listItem.appendChild(option);
        runtimeSelect.appendChild(listItem);
    }
});

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
    let fileName = '';
    let isProcessing = false;
    document.getElementById('fileInput').addEventListener('change', async (event) => {
        if (isProcessing) {
            console.log("Processing is already in progress.");
            return; // Early exit if a process is already running
        }
        isProcessing = true;
        let fileInput = document.getElementById('fileInput');
        let file = fileInput.files[0];

        if (file && (file.name.endsWith('.csv') || file.name.endsWith('.xlsx'))) {
            selectedFile = file;
            let reader = new FileReader();
            reader.onload = async (e) => {
                let arrayBuffer = e.target.result;
                let data = new Uint8Array(arrayBuffer);

                await pyodideReady;
                fileName = file.name;
                pyodideInstance.FS.writeFile(fileName, data);

                // Load the file into pyodide as a pandas dataframe
                let pythonCode = `
import pandas as pd

try:
    if '${fileName}'.endswith('.csv'):
        df = pd.read_csv('${fileName}')
    else:
        df = pd.read_excel('${fileName}', engine='openpyxl')
except Exception as e:
    raise Exception("Unable to process file. Please ensure it is a CSV or a supported Excel format.")

df.columns.tolist()
                `;
                let columnNames = await pyodideInstance.runPython(pythonCode);

                // Check which columns do not consist entirely of numerical values
                pythonCode = `
import numpy as np

def is_numeric(col):
    return np.issubdtype(df[col].dtype, np.number)

[col for col in df.columns if not is_numeric(col)]
                `;
                let nonNumericColumns = await pyodideInstance.runPython(pythonCode);

                // Populate the runtime and column selection dropdowns
                // populateRuntimeSelect();
                const runtimeToggle = document.getElementById('runtimeToggle');
                runtimeToggle.textContent = initialRuntimeText;
                runtimeToggle.setAttribute('data-value', '');
                populateColumnNamesSelect(columnNames, nonNumericColumns);

                // Show the form container
                document.getElementById('formContainer').style.display = 'block';
            };
            reader.readAsArrayBuffer(file);
        } else {
            if(file) {
                alert('Please upload a CSV or Excel file.');
                fileName = '';
                resetForm();
            }
            else {
                alert('Please choose a file to upload.');
                fileName = '';
                resetForm();
            }
        }
        isProcessing = false
    });

    document.getElementById('submitBtn').addEventListener('click', async (event) => {
        if (isProcessing) {
            console.log("Processing is already in progress.");
            return; // Early exit if a process is already running
        }
        isProcessing = true;
        event.preventDefault(); // Prevent the default form submission

        // let runtime = runtimeSelect.value;
        // let runtime = document.getElementById('runtimeToggle').textContent;
        let runtime = document.getElementById('runtimeToggle').getAttribute('data-value');
        let categoricalColumns = Array.from(columnNamesSelect.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        // let categoricalColumns = Array.from(columnNamesSelect.selectedOptions).map(option => option.value);

        if (runtime) {
            disableInputs();
            document.getElementById('processingMessage').style.display = 'block'; // Show processing message
            // document.querySelector('.processing-message').style.display = 'block';
            // alert("HERE")
            // resetInputs()
        
            let pythonCode = `
from scipy.linalg import qr, qr_update
from sklearn.linear_model import LinearRegression

import time

# minutes = 1
# categorical_columns = []
minutes = ${parseInt(runtime)}  # Runtime from the select dropdown
categorical_columns = ${JSON.stringify(categoricalColumns)}  # Categorical columns from the textarea input

# Use minutes and categorical_columns in your Python code as needed
print(f"Runtime: {minutes} minutes")
print(f"Categorical Columns: {categorical_columns}")


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
            setTimeout(async () => {
                let output = await pyodideInstance.runPython(pythonCode);

                // Handle the output
                // document.getElementById('processing-message').style.display = 'none'; // Hide the processing message
                // resetInputs();
                createDownloadLink(output, fileName);
                isProcessing = false;
            }, 100); // Introduce a delay before running the Python code
            // enableInputs();
        } else {
            alert('Please select a runtime.');
            isProcessing = false;
            // resetInputs(); 
            // enableInputs();
        }
    });

    // function populateColumnNamesSelect(columnNames, nonNumericColumns) {
    //     const columnNamesSelect = document.getElementById('columnNamesSelect');
    //     columnNamesSelect.innerHTML = ''; // Clear existing options

    //     columnNames.forEach(column => {
    //         const listItem = document.createElement('li');
    //         const option = document.createElement('a');
    //         option.classList.add('dropdown-item', 'negative-py-1');
    //         option.href = '#';

    //         const checkbox = document.createElement('input');
    //         checkbox.type = 'checkbox';
    //         checkbox.value = column;
    //         checkbox.checked = nonNumericColumns.includes(column);
            
    //         if (nonNumericColumns.includes(column)) {
    //             checkbox.disabled = true;
    //         }
    //         const label = document.createElement('span');
    //         label.textContent = column;

    //         option.appendChild(checkbox);
    //         option.appendChild(label);
    //         listItem.appendChild(option);
    //         columnNamesSelect.appendChild(listItem);
    //     });
    // }

    function populateColumnNamesSelect(columnNames, nonNumericColumns) {
        const columnNamesSelect = document.getElementById('columnNamesSelect');
        columnNamesSelect.innerHTML = ''; // Clear existing options
        
        columnNames.forEach(column => {
            const listItem = document.createElement('li');
            const option = document.createElement('a');
            option.classList.add('dropdown-item', 'negative-py-1');
            option.href = '#';
        
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = column;
            checkbox.checked = nonNumericColumns.includes(column);
            
            if (nonNumericColumns.includes(column)) {
                checkbox.disabled = true; // Disable non-numeric columns
            }
        
            const label = document.createElement('span');
            label.textContent = column;
            
            checkbox.addEventListener('click', function (event) {
                event.stopPropagation(); // Prevent the click event from propagating to the option element
            });

            // Add click event listener to the option element
            option.addEventListener('click', function (event) {
                event.preventDefault(); // Prevent the default behavior of the anchor tag
                event.stopPropagation(); // Prevent the click event from propagating to the dropdown
                if (!checkbox.disabled) {
                    checkbox.checked = !checkbox.checked; // Toggle the checkbox state if it's not disabled
                }
            });
        
            option.appendChild(checkbox);
            option.appendChild(label);
            listItem.appendChild(option);
            columnNamesSelect.appendChild(listItem);
        });
    }

    function createDownloadLink(csvData, processed_fileName) {
        const dataUrl = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvData);
        sessionStorage.setItem('downloadUrl', dataUrl);
        sessionStorage.setItem('fileName', 'processed_' + processed_fileName);
        window.location.href = 'download.html';
    }

    // function resetInputs() {
    //     document.getElementById('fileInput').value = '';
    //     document.getElementById('runtimeSelect').value = '';
    //     document.getElementById('columnNamesSelect').value = '';
    //     // Additional UI adjustments if necessary
    // }

    function disableInputs() {
        document.getElementById('submitBtn').disabled = true;
        document.getElementById('fileInput').disabled = true;
        document.getElementById('runtimeSelect').disabled = true;
        document.getElementById('columnNamesSelect').disabled = true;
    }

    function resetForm() {
        document.getElementById('fileInput').value = ''; // Clear the file input
        document.getElementById('runtimeToggle').textContent = 'Select Desired Runtime (mins)'; // Reset the runtime dropdown text
        document.getElementById('runtimeToggle').setAttribute('data-value', ''); // Reset the runtime dropdown data-value
        document.getElementById('columnNamesSelect').innerHTML = ''; // Clear the column select options
        document.getElementById('formContainer').style.display = 'none'; // Hide the form container
    }

    // function enableInputs() {
    //     // List of elements to re-enable
    //     const elements = [
    //         document.getElementById('fileInput'),
    //         document.getElementById('runtimeSelect'),
    //         document.getElementById('columnNamesSelect'),
    //         document.getElementById('submitBtn')
    //     ];

    //     elements.forEach(element => {
    //         setTimeout(() => {
    //             element.disabled = false;
    //         }, 100); // Delay re-enabling to ensure queued events are not executed
    //     });
    // }
};
