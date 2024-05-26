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
            
            setTimeout(async () => {
                const pythonScriptUrl = 'discrepancy_algos.py'; // Specify the path to your Python script
                fetch(pythonScriptUrl)
                    .then(response => response.text())
                    .then(async (pythonScript) => {
                        // Inject runtime and categoricalColumns into the Python script
                        pythonScript = pythonScript.replace('PLACEHOLDER_RUNTIME', parseInt(runtime))
                                                .replace('PLACEHOLDER_CATEGORICAL_COLUMNS', JSON.stringify(categoricalColumns))
                                                .replace('PLACEHOLDER_INCLUDED_COLUMNS', JSON.stringify(includedColumns))
                                                .replace('PLACEHOLDER_NORMALIZE', normalizeData ? 'True' : 'False');
                        let [output, times, discrepancy_values] = await pyodideInstance.runPython(pythonScript);
                        sessionStorage.setItem('xValues', JSON.stringify(Array.from(times)));
                        sessionStorage.setItem('yValues', JSON.stringify(Array.from(discrepancy_values)));
                        // Handle the output
                        createDownloadLink(output, fileName);              
                    })
                    .catch(error => {
                        console.error('Failed to run Python script:', error);
                        alert('An error occurred with the inputted file. Your file may be too large or in an incorrect format. Please refresh the page and try again.');
                    });
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
        document.getElementById('runtimeToggle').textContent = initialRuntimeText; // Reset the runtime dropdown text
        document.getElementById('runtimeToggle').setAttribute('data-value', ''); // Reset the runtime dropdown data-value
        document.getElementById('categoricalColumnSelect').innerHTML = ''; // Clear the column select options
        document.getElementById('columnSelectDropdown').innerHTML = ''; // Clear the column select options
        document.getElementById('normalizeToggle').checked = true;
        document.getElementById('formContainer').style.display = 'none'; // Hide the form container
    }
};
