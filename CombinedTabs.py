## Authored by Sebastian Torres

# All imports, which rely on PyQT and the functions present in the repo
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QLineEdit, QComboBox, QMessageBox, QProgressBar, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from plot_helpers import pre_processed_plots
from learning_utils import err_analysis, get_dataLoader, data_test, train_model
import pandas as pd
import numpy as np
from preprocessing_helpers import minMax, get_smoothed_data, get_LPF_filtered_data
from os import listdir
import torch
from torch.utils.data import DataLoader
from lstm_helpers import get_model_training_helpers


## Class to aid in training and display a progress bar
class TrainingThread(QThread):
    progress = pyqtSignal(int)
    completed = pyqtSignal()

    def __init__(self, model, trainloader, valloader, device, optimizer, loss_fn, epochs):
        super().__init__()
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs

    def run(self):
        for epoch in range(self.epochs):
            # Simulate training progress
            train_model(self.model, self.trainloader, self.valloader, self.device, self.optimizer, self.loss_fn, 1)
            self.progress.emit(int((epoch + 1) / self.epochs * 100))
        self.completed.emit()


## Main class for PyQT GUI Integration
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Engine Prediction GUI")
        self.setGeometry(100, 100, 500, 350)

        # Ensure all tabs and components are initialized
        self.progress_bar = None

        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_data_results = QWidget()
        self.tab_run_model = QWidget()

        self.tabs.addTab(self.tab_data_results, "Data and Results")
        self.tabs.addTab(self.tab_run_model, "Run Model")

        self.df_train = None
        self.df_test = None
        self.model = None
        self.training_thread = None

        self.init_data_results_tab()
        self.init_run_model_tab()


    ## Results Tab
    def init_data_results_tab(self):
        layout = QVBoxLayout()

        # Button to display pre-processed data plots
        self.btn_show_plots = QPushButton("Show Pre-processed Data Plots")
        self.btn_show_plots.clicked.connect(self.show_preprocessed_plots)
        layout.addWidget(self.btn_show_plots)

        # Button to run LSTM analysis
        self.btn_run_lstm = QPushButton("Run LSTM Analysis")
        self.btn_run_lstm.clicked.connect(self.run_lstm_analysis)
        layout.addWidget(self.btn_run_lstm)

        # Text area to display LSTM results
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        layout.addWidget(self.results_display)

        # Button to display LSTM analysis plots
        self.btn_show_lstm_plots = QPushButton("Show LSTM Analysis Plots")
        self.btn_show_lstm_plots.clicked.connect(self.show_lstm_plots)
        layout.addWidget(self.btn_show_lstm_plots)

        self.tab_data_results.setLayout(layout)


    ## Model Running Tab
    def init_run_model_tab(self):
        layout = QVBoxLayout()

        # Input for number of epochs
        self.epochs_input = QLineEdit()
        self.epochs_input.setPlaceholderText("Enter number of epochs (default: 100)")
        layout.addWidget(QLabel("Epochs:"))
        layout.addWidget(self.epochs_input)

        # Input for learning rate
        self.lr_input = QLineEdit()
        self.lr_input.setPlaceholderText("Enter learning rate (default: 1e-3)")
        layout.addWidget(QLabel("Learning Rate:"))
        layout.addWidget(self.lr_input)

        # Dropdown for loss function selection
        self.loss_fn_dropdown = QComboBox()
        self.loss_fn_dropdown.addItems(["nn.MSELoss", "nn.CrossEntropyLoss", "nn.L1Loss"])
        layout.addWidget(QLabel("Loss Function:"))
        layout.addWidget(self.loss_fn_dropdown)

        # Button to start training
        self.btn_train_model = QPushButton("Train Model")
        self.btn_train_model.clicked.connect(self.train_model_handler)
        layout.addWidget(self.btn_train_model)

        # Button to save the model
        self.btn_save_model = QPushButton("Save Model")
        self.btn_save_model.clicked.connect(self.save_model_handler)
        layout.addWidget(self.btn_save_model)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.tab_run_model.setLayout(layout)


    ## Workflow of this is taken from the main.py file
    def show_preprocessed_plots(self):

        col_names = [
            'Engine Unit',
            'time',
            'os1','os2','os3',
            'Fan Inlet Temp',                           #s1
            'LPC Outlet Temp',                          #s2
            'HPC Outlet Temp',                          #s3
            'LPT Outlet Temp',                          #s4
            'Fan Inlet Pressure',                       #s5
            'Bypass-Duct Pressure',                     #s6
            'HPC Outlet Pressure',                      #s7
            'Physical Fan Speed',                       #s8
            'Physical Core Speed',                      #s9
            'Engine Pressure Ratio (P50/P2)',           #s10
            'HPC Outlet Static Pressure',               #s11
            'Ratio of Fuel Flow to Ps30 (pps/psia)',    #s12
            'Corrected Fan Speed',                      #s13
            'Corrected Core Speed',                     #s14
            'Bypass Ratio',                             #s15
            'Burner Fuel-Air Ratio',                    #s16
            'Bleed Enthalpy',                           #s17
            'Required Fan Speed',                       #s18
            'Required Fan Conversion Speed',            #s19
            'High-Pressure Turbines Cool Air Flow',     #s20
            'Low-Pressure Turbines Cool Air Flow'       #s21
        ]

        drop_cols1 = [
        'os3',
        'Fan Inlet Temp',                           #s1
        'Fan Inlet Pressure',                       #s5
        'Bypass-Duct Pressure',                     #s6
        'Engine Pressure Ratio (P50/P2)',           #s10
        'Burner Fuel-Air Ratio',                    #s16
        'Required Fan Speed',                       #s18
        'Required Fan Conversion Speed',            #s19
        ]

        def get_rul_test_train(_df_test, _rul_test, _df_train):
            rul_list_train = []
            for n in np.arange(1,101):
                time_list = np.array(_df_train[_df_train['Engine Unit'] == n]['time'])
                length = len(time_list)
                rul = list(length - time_list)
                rul_list_train += rul
            rul_list_test = []

            for n in np.arange(1,101):
                time_list = np.array(_df_test[_df_test['Engine Unit'] == n]['time'])
                length = len(time_list)
                rul_val = _rul_test.iloc[n-1].item()
                rul = list(length - time_list + rul_val)
                rul_list_test += rul
            return rul_list_test, rul_list_train
        
        # LOAD DATA (loads every dataset even though they're not all used in GUI for future implementation)
        folder_path = './CMAPS/'
        listdir(folder_path)
        df_train_fd001 = pd.read_csv(folder_path + 'train_FD001.txt', header = None, sep = ' ')
        df_train_fd002 = pd.read_csv(folder_path + 'train_FD002.txt', header = None, sep = ' ')
        df_train_fd003 = pd.read_csv(folder_path + 'train_FD003.txt', header = None, sep = ' ')
        df_train_fd004 = pd.read_csv(folder_path + 'train_FD004.txt', header = None, sep = ' ')
        df_test_fd001 = pd.read_csv(folder_path + 'test_FD001.txt', header = None, sep = ' ')
        df_test_fd002 = pd.read_csv(folder_path + 'test_FD002.txt', header = None, sep = ' ')
        df_test_fd003 = pd.read_csv(folder_path + 'test_FD003.txt', header = None, sep = ' ')
        df_test_fd004 = pd.read_csv(folder_path + 'test_FD004.txt', header = None, sep = ' ')
        rul_test_fd001 = pd.read_csv(folder_path + 'RUL_FD001.txt', header = None)
        rul_test_fd002 = pd.read_csv(folder_path + 'RUL_FD002.txt', header = None)
        rul_test_fd003 = pd.read_csv(folder_path + 'RUL_FD003.txt', header = None)
        rul_test_fd004 = pd.read_csv(folder_path + 'RUL_FD004.txt', header = None)
        df_train_fd001 = df_train_fd001.iloc[:,:-2].copy()
        df_train_fd002 = df_train_fd002.iloc[:,:-2].copy()
        df_train_fd003 = df_train_fd003.iloc[:,:-2].copy()
        df_train_fd004 = df_train_fd004.iloc[:,:-2].copy()
        df_train_fd001.columns = col_names
        df_train_fd002.columns = col_names
        df_train_fd003.columns = col_names
        df_train_fd004.columns = col_names
        df_test_fd001 = df_test_fd001.iloc[:,:-2].copy()
        df_test_fd002 = df_test_fd002.iloc[:,:-2].copy()
        df_test_fd003 = df_test_fd003.iloc[:,:-2].copy()
        df_test_fd004 = df_test_fd004.iloc[:,:-2].copy()
        df_test_fd001.columns = col_names
        df_test_fd002.columns = col_names
        df_test_fd003.columns = col_names
        df_test_fd004.columns = col_names



        ##############################
        #   Dataset & Preprocessing
        ##############################

        # Get data frames & rul lists
        df_test  = df_test_fd001
        df_train = df_train_fd001
        rul_test = rul_test_fd001
        rul_list_test, rul_list_train = get_rul_test_train(df_test, rul_test, df_train)
        df_test['rul'], df_train['rul'] = rul_list_test, rul_list_train

        # Chose sample engine
        # sample_df = df_train[df_train['Engine Unit'] == SAMPLE].copy()

        # 1) Chose features
        df_train = df_train.drop(drop_cols1, axis = 1)
        df_test = df_test.drop(drop_cols1, axis = 1)

        # 2) MinMax Scaling
        df_train = minMax(df_train)
        df_test  = minMax(df_test)

        self.df_train = df_train
        self.df_test = df_test

        # 3) Smoothing: Exponentially Weighted Average
        df_train_smoothed, df_test_smoothed = get_smoothed_data(df_train, df_test)

        # 4) Low Pass Filter
        df_train_LPF, df_test_LPF = get_LPF_filtered_data(df_train, df_test, cutoff_low=12, fs=1000, order=5)

        # 5) Preprocessing
        sample = 10
        sample_df               = df_train[df_train['Engine Unit'] == sample].copy()
        smoothed_sample_df      = df_train_smoothed[df_train_smoothed['Engine Unit'] == sample].copy()
        LPF_sample_df           = df_train_LPF[df_train_LPF['Engine Unit'] == sample].copy()

        # Sample Data
        samples = [sample_df,smoothed_sample_df,LPF_sample_df]
        labels = ['original','smoothed','LPF']
        pre_processed_plots(samples, labels)
        plt.show()  # Display the plots


    ## Function to run the LSTM test and grab the error values
    def run_lstm_analysis(self):
        if self.df_train is None or self.df_test is None:
            self.results_display.setText("Please run preprocessing first.")
            return


        # LSTM Analysis Workflow
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        n_features = len(self.df_train.columns[2:-1])
        window = 20
        np.random.seed(5)
        units = np.arange(1, 101)
        train_units = list(np.random.choice(units, 80, replace=False))
        val_units = list(set(units) - set(train_units))

        train_data = self.df_train[self.df_train['Engine Unit'].isin(train_units)].copy()
        val_data = self.df_train[self.df_train['Engine Unit'].isin(val_units)].copy()
        trainloader, valloader = get_dataLoader(train_data, val_data, self.df_train, window)
        test = data_test(units, self.df_test)
        testloader = DataLoader(test, batch_size=100)
        model, loss_fn, optimizer = get_model_training_helpers(n_features, device)

        model.load_state_dict(torch.load('model.pth'))

        mse, std_dev, se, confidence_interval = err_analysis(model, loss_fn, testloader, device, _name='original', show=False)

        results = (f"Mean Squared Error: {mse}\n"
                   f"Standard Deviation: {std_dev}\n"
                   f"Standard Error: {se}\n"
                   f"Confidence Interval: {confidence_interval}")
        self.results_display.setText(results)

        # Creating and saving the model in the GUI framework to allow ability to run
        self.model, _, _ = get_model_training_helpers(n_features, device)


    ## Function to train the model directly in the GUI
    def train_model_handler(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Model is not initialized. Please run LSTM analysis first.")
            return

        # Get user inputs
        epochs = int(self.epochs_input.text()) if self.epochs_input.text() else 100
        learning_rate = float(self.lr_input.text()) if self.lr_input.text() else 1e-3


        # Define the loss function based on user choice
        if self.loss_fn_dropdown.currentText() == "nn.MSELoss":
            loss_fn = torch.nn.MSELoss()
        elif self.loss_fn_dropdown.currentText() == "nn.CrossEntropyLoss":
            loss_fn = torch.nn.CrossEntropyLoss()
        elif self.loss_fn_dropdown.currentText() == "nn.L1Loss":
            loss_fn = torch.nn.L1Loss()
        else:
            None

        if loss_fn is None:
            QMessageBox.warning(self, "Warning", "Invalid loss function selected.")
            return

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Assuming trainloader and valloader are prepared in run_lstm_analysis
        trainloader, valloader = get_dataLoader(self.df_train, self.df_test, self.df_train, 20)

        # Start training in a separate thread
        self.training_thread = TrainingThread(self.model, trainloader, valloader, device, optimizer, loss_fn, epochs)
        self.training_thread.progress.connect(self.progress_bar.setValue)
        self.training_thread.completed.connect(lambda: QMessageBox.information(self, "Info", "Model training completed."))
        self.training_thread.start()


    ## Function to export the trained model to a .pth file
    def save_model_handler(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model available to save. Train the model first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "PyTorch Model (*.pth)")
        if file_path:
            torch.save(self.model.state_dict(), file_path)
            QMessageBox.information(self, "Info", f"Model saved successfully at {file_path}.")


    ## Function to show the results in plot format
    def show_lstm_plots(self):
        if self.df_train is None or self.df_test is None:
            self.results_display.setText("Please run preprocessing first.")
            return

        if self.df_train is None or self.df_test is None:
            self.results_display.setText("Please run preprocessing first.")
            return

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        n_features = len(self.df_train.columns[2:-1])
        window = 20
        np.random.seed(5)
        units = np.arange(1, 101)
        test = data_test(units, self.df_test)
        testloader = DataLoader(test, batch_size=100)
        model, loss_fn, _ = get_model_training_helpers(n_features, device)
        model.load_state_dict(torch.load('model.pth'))
        err_analysis(model, loss_fn, testloader, device, _name='original', show=True)


## Main function to run the GUI
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())