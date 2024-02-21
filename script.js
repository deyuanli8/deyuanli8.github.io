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
            disableInputs()
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

                    // Fetch and run Python code from an external file
                    const pythonScriptUrl = 'discrepancy_finder.py'; // Specify the path to your Python script
                    fetch(pythonScriptUrl)
                        .then(response => response.text())
                        .then(async (pythonScript) => {
                            // Inject runtime and categoricalColumns into the Python script
                            pythonScript = pythonScript.replace('PLACEHOLDER_RUNTIME', parseInt(runtime))
                                                       .replace('PLACEHOLDER_CATEGORICAL_COLUMNS', JSON.stringify(categoricalColumns))
                                                       .replace('PLACEHOLDER_FILENAME', fileName);
                            let output = await pyodideInstance.runPython(pythonScript);

                            // Process the output as needed
                            processingMessage.style.display = 'none'; // Hide processing message
                            resetInputs();
                            createDownloadLink(output, fileName);
                            enableInputs();
                            // Reset inputs after processing
                            
                        })
                        .catch(error => console.error('Failed to run Python script:', error));
                };
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
