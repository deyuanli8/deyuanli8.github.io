# Discrepancy Solver Website

This website allows users to upload an Excel or CSV file, run a discrepancy algorithm on the data, and download the resulting file with an additional column indicating the group assignment for each row. The application also generates a plot based on the discrepancy algorithm results to show discrepancy values over time, which is displayed on the download page.

## Setup and Usage

1. Click on the "Choose File" button to select an Excel or CSV file to upload. Make sure that each column has a header.
2. Once the file is uploaded, the application will automatically detect the column names and generate dropdown menus.
3. Select the desired runtime from the first dropdown menu.
4. Use the normalization toggle to switch normalization on or off.
5. In the second dropdown menu, select the columns to include in the discrepancy algorithm calculation by toggling the checkboxes.
6. In the third dropdown menu, select the columns to treat as categorical variables. Columns with non-numeric values will be toggled on by default and cannot be changed.
7. Click the "Submit" button to perform the discrepancy algorithm on the input data.
8. The application will generate a new CSV file with an extra column named "Group" indicating the group assignment for each row.
9. The application will redirect to `download.html`, where the plot showing the best discrepancy values found by the algorithm over time is displayed.
10. Click on the "Download Processed File" button to download the generated CSV file containing the group assignment.
11. To restart the process with a new file or different parameters, click the "Restart" button on the download page.

## Discrepancy Definition
We mathematically describe the problem we are trying to solve. Suppose there are $n$ people and $m$ characteristics per person. If each characteristic is represented by a numeric val, then for all $1 \le i \le n$, let $v_i \in \mathbb{R}^m$ be the vector corresponding to the $m$ characteristic values for person $i$. Then we wish to find a coloring or split $x \in \{-1, 1\}^n$ that minimizes the discrepancy value:
$$\Vert Vx\Vert_\infty = \left \Vert \sum_{i=1}^n x(i)v_i \right \Vert_\infty.$$ 
Here, $V = [v_1, \ldots, v_n] \in \mathbb{R}^{m \times n}$. Then person $i$ is assigned to Group A if $x(i) = -1$, and it is assigned to Group B otherwise.

In our setup, each row of the the inputted Excel or CSV file corresponds to the characteristics for a singular person. Each column corresponds to a single charactersitic's values over all people. When normalization is applied, every numeric value is scaled so that the smallest value is $-1$ and the largest value is $1$. This helps prevent extreme weighting in the optimization objective toward a singular characteristic. When columns corresponding to categorical characteristics are used in the discrepancy calculation, if the column consists of $k$ different possible categories, then the column is replaced with $k$ different columns corresponding to the dummy variable for one of the classes. If normalization is applied, the dummy variable transformation is applied first and then normalization is applied. The website applies many heuristics, including different reduction techniques (if $n > m$) and numerous local search techniques, to find a split.

## Features

- Allows users to upload an Excel or CSV file containing columns corresponding to each row's characteristics
- Automatically detects column names and generates dropdown menus for runtime selection, column inclusion, and categorical variable selection
- Performs a discrepancy algorithm on the input data to split rows into two groups
- Displays a plot on the download page showing the best discrepancies found over time
- Allows users to download the resulting file with an extra column indicating the group assignment
- Allows users to restart the process with a new file or different parameters
- No servers are needed, since all code is run client-side

## Project Structure

The project consists of the following files:

- `index.html`: The main page where users can upload the file and select parameters
- `download.html`: The page where users can download the resulting file and view the line graph
- `script.js`: The JavaScript file containing the logic for handling file upload, running the discrepancy algorithm using Pyodide, and generating the line graph
- `discrepancy_algos.py`: The file containing the bulk of the Python code that is fetched by the script to be run through Pyodide. 

*This website was created by Deyuan Li under the guidance of Professor Daniel Spielman. LLM's were used to help aid in the construction of this website.*