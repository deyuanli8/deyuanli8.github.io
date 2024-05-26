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
            return;
        }
        isProcessing = true;
        let fileInput = document.getElementById('fileInput');
        let file = fileInput.files[0];

        if (file && (file.name.endsWith('.csv') || file.name.endsWith('.xlsx'))) {
            let reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    let arrayBuffer = e.target.result;
                    let data = new Uint8Array(arrayBuffer);

                    document.getElementById('loadingMessage').style.display = 'block';

                    await pyodideReady;
                    fileName = file.name;
                    pyodideInstance.FS.writeFile(fileName, data);

                    let columnNames = await pyodideInstance.runPython(`
                        import pandas as pd
                        try:
                            df = pd.read_csv('${fileName}') if '${fileName}'.endswith('.csv') else pd.read_excel('${fileName}', engine='openpyxl')
                        except Exception as e:
                            raise Exception("Unable to process file. Please ensure it is a CSV or a supported Excel format.")
                        df.columns.tolist()
                    `);

                    let nonNumericColumns = await pyodideInstance.runPython(`
                        import numpy as np
                        df = pd.read_csv('${fileName}') if '${fileName}'.endswith('.csv') else pd.read_excel('${fileName}', engine='openpyxl')
                        [col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)]
                    `);

                    document.getElementById('loadingMessage').style.display = 'none';
                    const runtimeToggle = document.getElementById('runtimeToggle');
                    runtimeToggle.textContent = initialRuntimeText;
                    runtimeToggle.setAttribute('data-value', '');
                    selectedCategoricalColumns = nonNumericColumns.slice();

                    document.getElementById('normalizeToggle').checked = true;
                    populateCategoricalColumnSelect(columnNames, nonNumericColumns, nonNumericColumns);
                    populateColumnSelectDropdown(columnNames, nonNumericColumns);

                    document.getElementById('formContainer').style.display = 'block';
                } catch (error) {
                    console.error('Error processing file:', error);
                    alert('Failed to process the file. Please check the format and try again.');
                    fileName = '';
                    resetForm();
                }
            };
            reader.onerror = () => {
                alert('Error reading file. Please ensure the file is not corrupted.');
                fileName = '';
                resetForm();
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
        isProcessing = false;
    });

    document.getElementById('submitBtn').addEventListener('click', async (event) => {
        if (isProcessing) {
            console.log("Processing is already in progress.");
            return;
        }
        isProcessing = true;
        event.preventDefault();

        let runtime = document.getElementById('runtimeToggle').getAttribute('data-value');
        let categoricalColumns = Array.from(document.getElementById('categoricalColumnSelect').querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        let includedColumns = Array.from(document.getElementById('columnSelectDropdown').querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        let normalizeData = document.getElementById('normalizeToggle').checked;

        if (runtime && includedColumns.length > 0) {
            disableInputs();
            document.getElementById('processingMessage').style.display = 'block';

            setTimeout(async () => {
                try {
                    const pythonScriptUrl = 'discrepancy_algos.py';
                    let response = await fetch(pythonScriptUrl);
                    let pythonScript = await response.text();
                    pythonScript = pythonScript.replace('PLACEHOLDER_RUNTIME', parseInt(runtime))
                                                .replace('PLACEHOLDER_CATEGORICAL_COLUMNS', JSON.stringify(categoricalColumns))
                                                .replace('PLACEHOLDER_INCLUDED_COLUMNS', JSON.stringify(includedColumns))
                                                .replace('PLACEHOLDER_NORMALIZE', normalizeData ? 'True' : 'False');

                    let [output, times, discrepancy_values] = await pyodideInstance.runPython(pythonScript);
                    sessionStorage.setItem('xValues', JSON.stringify(Array.from(times)));
                    sessionStorage.setItem('yValues', JSON.stringify(Array.from(discrepancy_values)));
                    createDownloadLink(output, fileName);
                } catch (error) {
                    console.error('Failed to run Python script:', error);
                    alert('An error occurred with the inputted file. Your file may be too large or in an incorrect format. Please refresh the page and try again.');
                    resetForm();
                }
                isProcessing = false;
            }, 100);
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
