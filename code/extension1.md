To run this file, we uploaded the notebook to Google Colab and connected to and A100 GPU runtime. Then run all the cells up to and including
the one that trains the model to get a baseline model. Running for 20 epochs as is currently in the notebook took about 4 hours, so if you
just want to test it without waiting this long you can change the number of epochs in this cell. If you want to download the saved model,
you can run the next two cells to zip the output folder and download it. If you want to run the experiment with our model trained on 20
epochs, you can download the zip file from the following Google Drive link: 

https://drive.google.com/file/d/1Souaej0KMwFtTfW21_ROwjIrk6WUBI0U/view?usp=sharing

Then upload the zip file to the Google Colab files and run the next cell to unzip it. To test the results for both the baseline and with
the extension applied, run the remaining cells.

Note: When running in Colab, you also need to upload the `data/` folder from this repo (it contains `train.csv`, `dev.csv`, and `test.csv`).