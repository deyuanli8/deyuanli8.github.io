let pyodideInstance = null;
let initialRuntimeText = '';
let selectedCategoricalColumns = [];

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
            let reader = new FileReader();
            reader.onload = async (e) => {
                let arrayBuffer = e.target.result;
                let data = new Uint8Array(arrayBuffer);
                
                document.getElementById('loadingMessage').style.display = 'block'; // Show the loading message

                await pyodideReady;
                fileName = file.name;
                pyodideInstance.FS.writeFile(fileName, data);

                // Load the file into pyodide as a pandas dataframe
                let pythonCodeLoadDf = `
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
                let columnNames = await pyodideInstance.runPython(pythonCodeLoadDf);
                // Check which columns do not consist entirely of numerical values
                let pythonCodeFindCategorical = `
import numpy as np

def is_numeric(col):
    return np.issubdtype(df[col].dtype, np.number)

[col for col in df.columns if not is_numeric(col)]
                `;
                let nonNumericColumns = await pyodideInstance.runPython(pythonCodeFindCategorical);


                document.getElementById('loadingMessage').style.display = 'none'; // Hide the loading message
                const runtimeToggle = document.getElementById('runtimeToggle');
                runtimeToggle.textContent = initialRuntimeText;
                runtimeToggle.setAttribute('data-value', '');
                selectedCategoricalColumns = nonNumericColumns.slice();

                document.getElementById('normalizeToggle').checked = true;
                populateCategoricalColumnSelect(columnNames, nonNumericColumns, nonNumericColumns);
                populateColumnSelectDropdown(columnNames, nonNumericColumns)

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

        let runtime = document.getElementById('runtimeToggle').getAttribute('data-value');
        let categoricalColumns = Array.from(categoricalColumnSelect.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        let includedColumns = Array.from(columnSelectDropdown.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        let normalizeData = document.getElementById('normalizeToggle').checked;

        if (runtime && includedColumns.length > 0) {
            disableInputs();
            document.getElementById('processingMessage').style.display = 'block'; // Show processing message
        
            let pythonCodeCalculateDiscrepancy = `
from scipy.linalg import qr, qr_update
from sklearn.linear_model import LinearRegression

import time

minutes = ${parseInt(runtime)}  # Runtime from the select dropdown
categorical_columns = ${JSON.stringify(categoricalColumns)}  # Categorical columns from the textarea input
included_columns = ${JSON.stringify(includedColumns)} # Columns to be included in discrepancy calculation
normalize = ${normalizeData ? 'True' : 'False'} # Whether we should normalize data

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

def get_split(df_vecs, minutes = 5, categorical_columns = None, included_columns = None, normalize = True):
    def find_best(df_current, current_discrepancy, df_new):
        nonlocal times, discrepancy_values 
        new_discrepancy = compute_discrepancy(df_new, disp = False)
        if current_discrepancy <= new_discrepancy:
            return (df_current, current_discrepancy)
        print("Better Discrepancy Found:", new_discrepancy)
        times.append((time.time() - start_time)/60)
        discrepancy_values.append(new_discrepancy)
        return (df_new, new_discrepancy)
    start_time = time.time()
    df = df_vecs.copy()
    times = []
    discrepancy_values = []

    if included_columns is None:
        included_columns = df.columns
    if not set(included_columns).issubset(set(df.columns)):
        print("Error: Inputted list of included columns are not part of dataframe.")
        return None 
    if categorical_columns is not None:
        if not set(categorical_columns).issubset(set(included_columns)):
            print("Error: Inputted list of categorical columns are not part of included columns.")
            return None

    df_encoded = pd.get_dummies(df[included_columns], columns=categorical_columns).astype(float)
    df_encoded.columns = [f'Characteristic_{i}' for i in range(len(df_encoded.columns))]
    m = len(df_encoded.columns)
    if normalize:
        df_reduced = faster_reduction(normalize_data(df_encoded))
    else:
        df_reduced = faster_reduction(df_encoded)

    df_best = local_search(df_reduced, 4, cols = df_reduced[abs(df_reduced['x'])!=1].index)
    best_discrepancy = compute_discrepancy(df_best, disp = False)
    print("Starting Discrepancy:", best_discrepancy)
    times.append((time.time() - start_time)/60)
    discrepancy_values.append(best_discrepancy)

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
    times.append((time.time() - start_time)/60)
    discrepancy_values.append(best_discrepancy)
    print("Final Discrepancy:", best_discrepancy)

    df['Group'] = np.where(df_best['x'] > 0, 'A', 'B')
    return df, times, discrepancy_values

df_output, times, discrepancy_values = get_split(df, minutes, categorical_columns, included_columns, normalize)


output = df_output.to_csv(index=False)
output, times, discrepancy_values
            `;
            setTimeout(async () => {
                let [output, times, discrepancy_values] = await pyodideInstance.runPython(pythonCodeCalculateDiscrepancy);

                sessionStorage.setItem('xValues', JSON.stringify(Array.from(times)));
                sessionStorage.setItem('yValues', JSON.stringify(Array.from(discrepancy_values)));
                // Handle the output
                createDownloadLink(output, fileName);
                isProcessing = false;
            }, 100); // Introduce a delay before running the Python code
        } else {
            if(!runtime){
                alert('Please select a runtime.');
            }
            else{
                alert('Please select at least one column to include in the discrepancy calculation.');
            }
            isProcessing = false;
        }
    });


    function populateCategoricalColumnSelect(selectedColumns, categoricalColumns, nonNumericColumns) {
        const categoricalColumnSelect = document.getElementById('categoricalColumnSelect');
        categoricalColumnSelect.innerHTML = ''; // Clear existing options
        
        selectedColumns.forEach(column => {
            const listItem = document.createElement('li');
            const option = document.createElement('a');
            option.classList.add('dropdown-item', 'negative-py-1');
            option.href = '#';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = column;
            checkbox.checked = categoricalColumns.includes(column);
            
            if (nonNumericColumns.includes(column)) {
                checkbox.disabled = true; // Disable non-numeric columns
            }
            
            const label = document.createElement('span');
            label.textContent = column;

            checkbox.addEventListener('click', function (event) {
                event.stopPropagation(); // Prevent the click event from propagating to the option element
                if (checkbox.checked) {
                    selectedCategoricalColumns.push(column);
                } else {
                    const index = selectedCategoricalColumns.indexOf(column);
                    if (index > -1) {
                        selectedCategoricalColumns.splice(index, 1);
                    }
                }
            });

            // Add click event listener to the option element
            option.addEventListener('click', function (event) {
                event.preventDefault(); // Prevent the default behavior of the anchor tag
                event.stopPropagation(); // Prevent the click event from propagating to the dropdown
                if (!checkbox.disabled) {
                    checkbox.checked = !checkbox.checked; // Toggle the checkbox state if it's not disabled
                    if (checkbox.checked) {
                        selectedCategoricalColumns.push(column);
                    } else {
                        const index = selectedCategoricalColumns.indexOf(column);
                        if (index > -1) {
                            selectedCategoricalColumns.splice(index, 1);
                        }
                    }
                }
            });
        
            option.appendChild(checkbox);
            option.appendChild(label);
            listItem.appendChild(option);
            categoricalColumnSelect.appendChild(listItem);
        });
    }


    function updateCategoricalColumns(nonNumericColumns) {
        const columnSelectDropdown = document.getElementById('columnSelectDropdown');
        const selectedColumns = Array.from(columnSelectDropdown.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        
        const categoricalColumns = selectedColumns.filter(column => selectedCategoricalColumns.includes(column));
        populateCategoricalColumnSelect(selectedColumns, categoricalColumns, nonNumericColumns);
    }


    function populateColumnSelectDropdown(columnNames, nonNumericColumns) {
        const columnSelectDropdown = document.getElementById('columnSelectDropdown');
        columnSelectDropdown.innerHTML = ''; // Clear existing options
        
        columnNames.forEach(column => {
            const listItem = document.createElement('li');
            const option = document.createElement('a');
            option.classList.add('dropdown-item', 'negative-py-1');
            option.href = '#';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = column;
            checkbox.checked = true; // Initially include all columns
            
            const label = document.createElement('span');
            label.textContent = column;
            
            checkbox.addEventListener('click', function (event) {
                event.stopPropagation(); // Prevent the click event from propagating to the option element
                updateCategoricalColumns(nonNumericColumns);
            });
            
            // Add click event listener to the option element
            option.addEventListener('click', function (event) {
                event.preventDefault(); // Prevent the default behavior of the anchor tag
                event.stopPropagation(); // Prevent the click event from propagating to the dropdown
                checkbox.checked = !checkbox.checked; // Toggle the checkbox state
                updateCategoricalColumns(nonNumericColumns);
            });
            
            option.appendChild(checkbox);
            option.appendChild(label);
            listItem.appendChild(option);
            columnSelectDropdown.appendChild(listItem);
        });
    }

    function createDownloadLink(csvData, processed_fileName) {
        const dataUrl = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvData);
        sessionStorage.setItem('downloadUrl', dataUrl);
        sessionStorage.setItem('fileName', 'processed_' + processed_fileName);
        window.location.href = 'download.html';
    }

    function disableInputs() {
        document.getElementById('submitBtn').disabled = true;
        document.getElementById('fileInput').disabled = true;
        document.getElementById('runtimeSelect').disabled = true;
        document.getElementById('categoricalColumnSelect').disabled = true;
        document.getElementById('columnSelectDropdown').disabled = true; // Clear the column select options
        document.getElementById('normalizeToggle').disabled = true;
    }

    function resetForm() {
        document.getElementById('fileInput').value = ''; // Clear the file input
        document.getElementById('runtimeToggle').textContent = 'Select Desired Runtime (mins)'; // Reset the runtime dropdown text
        document.getElementById('runtimeToggle').setAttribute('data-value', ''); // Reset the runtime dropdown data-value
        document.getElementById('categoricalColumnSelect').innerHTML = ''; // Clear the column select options
        document.getElementById('columnSelectDropdown').innerHTML = ''; // Clear the column select options
        document.getElementById('normalizeToggle').checked = true;
        document.getElementById('formContainer').style.display = 'none'; // Hide the form container
    }
};