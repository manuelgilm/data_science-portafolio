# Spam Detection

In this project I will show you how to use scikit-learn to implement a basic text classifier. The goal of this project is to classify text messages between spam and ham categories. The dataset can be downloaded from the following [link](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.
 
## Running the notebook.

### Using Anaconda environment.

To run the provided notebook you need to create a new anaconda environment. You can create a new environment with all the libraries required using the moviereview_env.yml file and run the following line:

    conda env create -f spam_detection.yml
    conda activate activate spam_detection
After activating the environment you can run the jupyter notebook.

### Using Docker.

You can use **Docker** to run the **Jupyter Notebook** directly, you only need to follow the next steps.

    docker pull gilsama/spam_detection_scikit_learn
    docker run -p 8888:8888 gilsama/spam_detection_scikit_learn
