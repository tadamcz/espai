## Guide to files
File names are generally taken from existing AI Impacts sources (in some cases with minor modifications). The directory structure is our own.


### `2023 ESPAI - Cleaned and Anonymized Responses.csv`
CSV copy of this [publicly available spreadsheet](https://docs.google.com/spreadsheets/d/1aOydfhZHuVwU_fwTgE0_O_-8p-uMrRDYV5R5QnwOMGI/)

### `combined-cleaned-personal-anon.csv`
Anonymized version of `combined-cleaned-personal.csv` from AI impacts sources. Created by dropping columns that contain personally identifiable information.

3270 individual responses.

This is the rawest form of the data we use in this repo. We can judge from the filename that it underwent some data cleaning from the raw data provided by Qualtrics. 

### `fields.csv`
Description of the column names. Incomplete: some fields are not described.

### `aggregated`
Contains the CDFs obtained from aggregating the responses of individual experts. 

1712 rows.

Fields:
- `x`
- `lower`: low end of ??% (TODO) bootstrap confidence interval for the y-value at x
- `upper`: high end of ??% (TODO) bootstrap confidence interval for the y-value at x
- `y`

### `gamma_fitted`
Contains the fitted Gamma CDFs, one for each individual response.

The most relevant fields are:
- `response.id`
- `shape`
- `scale`
- `convergence`
- `error` 