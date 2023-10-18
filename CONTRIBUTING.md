# RAGchain contribution guide
At first, thank you for your interest! RAGchain is a open-source project, so any contribution is welcome.
Any contribution like new features, fix bugs, better code, fix typo, etc. is welcome.

## How to contribute
To contribute, please fork this repository and send pull request. 
When you send pull request, please follow these guidelines.

### 1. Set clear Pull Request name
For clarify what was your problem, and what you do in this pull request is important.

### 2. Explain about your Contribution
You don't need to explain code line by line, but we want to know what you do for RAGchain project. Please write some explanation about your pull requests.

### 3. Write test code
If you made new feature, please write test code for that. 
Test code is located in our tests/RAGchain folder. You can add new test file or new test code at existed test .py file.

### 4. All Test Passed
Before you send pull request, please check all test passed. Because we want to know that your new feature or code eventually break something.
For testing, you have to install dev-requirements. Run following command:

```bash
pip install -r dev_requirements.txt
```

After installed, you have to make your pytest.ini. The template of this file in pytest.ini.
Then run following command for testing:

```bash
pytest
```

It is okay if you did not pass some tests that needs to run API server. Like hwp loader, nougat API Loader, etc.
If you couldn't run or pass that tests, please let us know when you send pull requests.
Plus, when you have some trouble with running tests, feel free to contact contributors.
