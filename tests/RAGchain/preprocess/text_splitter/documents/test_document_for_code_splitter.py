from langchain.schema import Document


class TEST_DOCUMENT_CODE_SPLITEER:
    # Here is test documents with all language that splitter can split for test.
    def __init__(
            self,
            PYTHON_TEST_DOCUMENT: Document = None,
            JS_TEST_DOCUMENT: Document = None,
            TS_TEST_DOCUMENT: Document = None,
            Markdown_TEST_DOCUMENT: Document = None,
            Latex_TEST_DOCUMENT: Document = None,
            HTML_TEST_DOCUMENT: Document = None,
            Solidity_TEST_DOCUMNET: Document = None,
            Csharp_TEST_DOCUMENT: Document = None,
    ):
        self.PYTHON_TEST_DOCUMENT = PYTHON_TEST_DOCUMENT
        self.JS_TEST_DOCUMENT = JS_TEST_DOCUMENT
        self.TS_TEST_DOCUMENT = TS_TEST_DOCUMENT
        self.Markdown_TEST_DOCUMENT = Markdown_TEST_DOCUMENT
        self.Latex_TEST_DOCUMENT = Latex_TEST_DOCUMENT
        self.HTML_TEST_DOCUMENT = HTML_TEST_DOCUMENT
        self.Solidity_TEST_DOCUMNET = Solidity_TEST_DOCUMNET
        self.Csharp_TEST_DOCUMENT = Csharp_TEST_DOCUMENT

    def PYTHON_TEST_DOCUMENT(self, PYTHON_TEST_DOCUMENT):
        if PYTHON_TEST_DOCUMENT is None:
            return Document(
                page_content="""
                            def hello_world():
                                print("Hello, World!")
                            
                            # Call the function
                            hello_world()
                                """,
                metadata={
                    'source': 'test_source',
                    # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
                    'Data information': 'test for python code document',
                    'Data reference link': 'https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter#python'
                }
            )
        else:
            return PYTHON_TEST_DOCUMENT

    def JS_TEST_DOCUMENT(self, JS_TEST_DOCUMENT):
        if JS_TEST_DOCUMENT is None:
            return Document(
                page_content="""
                                def hello_world():
                                    print("Hello, World!")
                                
                                # Call the function
                                hello_world()
                            """,
                metadata={
                    'source': 'test_source',
                    # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
                    'Data information': 'test for js code document',
                    'Data reference link': 'https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter#js',
                }
            )

        else:
            return JS_TEST_DOCUMENT

    def TS_TEST_DOCUMENT(self, TS_TEST_DOCUMENT):
        if TS_TEST_DOCUMENT is None:
            return Document(
                page_content="""
                                function helloWorld(): void {
                                  console.log("Hello, World!");
                                }
                                
                                // Call the function
                                helloWorld();
                            """,
                metadata={
                    'source': 'test_source',
                    # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
                    'Data information': 'test for js code document',
                    'Data reference link': 'https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter#js',
                }
            )

        else:
            return TS_TEST_DOCUMENT

    def Markdown_TEST_DOCUMENT(self, Markdown_TEST_DOCUMENT):
        if Markdown_TEST_DOCUMENT is None:
            return Document(
                page_content="""
                        # 🦜️🔗 욱박사님을 아세유?
                        
                        ⚡ 난 몰라유 그딴거 나는잘 몰라유 ⚡
                        
                        ## 맨까 새끼들 부들부들하구나~
                        
                        ```bash
                        # Hopefully this code block isn't split
                        pip install 엄랭귀
                        ```
                        
                        RAGchain SOTA!
                        """
                ,
                metadata={
                    'source': 'test_source',
                    # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
                    'Data information': 'test for markdown text document',
                    'Data reference link': 'https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter#markdown',
                }
            )

        else:
            return Markdown_TEST_DOCUMENT

    def Latex_TEST_DOCUMENT(self, Latex_TEST_DOCUMENT):
        if Latex_TEST_DOCUMENT is None:
            return Document(
                page_content="""
                            \documentclass{article}
                            
                            \begin{document}
                            
                            \maketitle
                            
                            \section{Introduction}
                            Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in a variety of natural language processing tasks, including language translation, text generation, and sentiment analysis.
                            
                            \subsection{History of LLMs}
                            The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.
                            
                            \subsection{Applications of LLMs}
                            LLMs have many applications in industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.
                            
                            \end{document}
                            """
                ,
                metadata={
                    'source': 'test_source',
                    # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
                    'Data information': 'test for latex text document',
                    'Data reference link': 'https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter#latex',
                }
            )

        else:
            return Latex_TEST_DOCUMENT

    def HTML_TEST_DOCUMENT(self, HTML_TEST_DOCUMENT):
        if HTML_TEST_DOCUMENT is None:
            return Document(
                page_content="""
                        <!DOCTYPE html>
                        <html>
                            <head>
                                <title>🦜️🔗 LangChain</title>
                                <style>
                                    body {
                                        font-family: Arial, sans-serif;
                                    }
                                    h1 {
                                        color: darkblue;
                                    }
                                </style>
                            </head>
                            <body>
                                <div>
                                    <h1>🦜️🔗 LangChain</h1>
                                    <p>⚡ Building applications with LLMs through composability ⚡</p>
                                </div>
                                <div>
                                    As an open-source project in a rapidly developing field, we are extremely open to contributions.
                                </div>
                            </body>
                        </html>
                """
                ,
                metadata={
                    'source': 'test_source',
                    # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
                    'Data information': 'test for html text document',
                    'Data reference link': 'https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter#html',
                }
            )

        else:
            return HTML_TEST_DOCUMENT

    def Solidity_TEST_DOCUMNET(self, Solidity_TEST_DOCUMNET):
        if Solidity_TEST_DOCUMNET is None:
            return Document(
                page_content="""
                        pragma solidity ^0.8.20;
                        contract HelloWorld {
                           function add(uint a, uint b) pure public returns(uint) {
                               return a + b;
                           }
                        }
                        """
                ,
                metadata={
                    'source': 'test_source',
                    # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
                    'Data information': 'test for solidity text document',
                    'Data reference link': 'https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter#solidity',
                }
            )

        else:
            return Solidity_TEST_DOCUMNET

    def Csharp_TEST_DOCUMENT(self, Csharp_TEST_DOCUMENT):
        if Csharp_TEST_DOCUMENT is None:
            return Document(
                page_content="""
                    using System;
                    class Program
                    {
                        static void Main()
                        {
                            int age = 30; // Change the age value as needed
                    
                            // Categorize the age without any console output
                            if (age < 18)
                            {
                                // Age is under 18
                            }
                            else if (age >= 18 && age < 65)
                            {
                                // Age is an adult
                            }
                            else
                            {
                                // Age is a senior citizen
                            }
                        }
                    }
                        """
                ,
                metadata={
                    'source': 'test_source',
                    # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
                    'Data information': 'test for C# text document',
                    'Data reference link': 'https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter#c',
                }
            )

        else:
            return Csharp_TEST_DOCUMENT
