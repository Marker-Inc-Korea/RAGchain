import pytest
from langchain.schema import Document

from RAGchain.preprocess.text_splitter import MarkDownHeaderSplitter

TEST_DOCUMENT = Document(
    page_content="""
    # yield가 뭐야?
    여타 함수는 return값을 함수 하나당 한개씩 반환할 수 있지만 yield 제너레이터는 return값을 여러번 나누어서 제공할 수 있다.
    (여러번 나누어서 제공한다는 것은 함수를 실행할때마다 yield한 순서대로 여러번을 리턴한다는것)
    
    예제코드
    ```python
    def yield_abc():
      yield "A"
      yield "B"
      yield "C"
      
    for ch in return_abc():
      print(ch)
      
    '''
    결과 값은
    A
    B
    C
    로 나온다.
    '''
    ```
    
    
    
    # Generateor가 뭐야?
    여러 개의 데이터를 미리 만들어 놓지 않고 필요할 때마다 즉석해서 하나씩 만들어낼 수 있는 객체를 의미.
    
    ## 왜써?
    반환 값이 여러개 즉석해서 하나씩 만들 수 있다면 
    결과값을 나누어서 얻을 수 있어 성능 측면에서 개이득이다.
    메모리 효율 측면에서 모든 결과 값을 return 키워드를 사용할 경우 모든 결과 값을 메모리 올려놓아야 하는 반면에, yield 키워드를 사용할 때는 결과 값을 하나씩 메모리에 올려놓습니다.
    
    이러한 이유 때문에 좀 더 효율적인 프로그램을 작성할 수 있는 경우가 많다.
    
    코딩테스트 할때 개꿀일듯?
    
    예제코드
    Case1)
    ```python
    import time
    
    def return_abc():
      alphabets = []
      for ch in "ABC":
        time.sleep(1)
        alphabets.append(ch)
      return alphabets
    ```
    ```python
    for ch in return_abc():
      print(ch)
    ```
    ```
    # 3초 경과
    A
    B
    C
    ```
    
    Case2)
    ```python
    import time
    
    def yield_abc():
      for ch in "ABC":
        time.sleep(1)
        yield ch
        
    for ch in yield_abc():
      print(ch)
      
    '''
    # 1초 경과
    A
    # 1초 경과
    B
    # 1초 경과
    C
    
    '''
    ```
    
    
    > 참고자료
    [1. 제너레이터 네이스한 설명](https://www.daleseo.com/python-yield/)
    """,
    metadata={
        'source': 'test_source'
    }
)


@pytest.fixture
def markdownheader_text_splitter():
    markdownheader_text_splitter = MarkDownHeaderSplitter()
    yield markdownheader_text_splitter


def test_markdownheader_text_splitter(markdownheader_text_splitter):
    passages, header_info = markdownheader_text_splitter.split_document(TEST_DOCUMENT)


    assert len(passages) > 1
    assert passages[0].next_passage_id == passages[1].id
    assert passages[1].previous_passage_id == passages[0].id
    assert passages[0].filepath == 'test_source'
    assert passages[0].filepath == passages[1].filepath
    assert passages[0].previous_passage_id is None
    assert passages[-1].next_passage_id is None

    for passages_num in range(len(passages)):
        # Check splitter preserve other metadata in original document.(original doc의 meta data에 여러개의 metadata가 있는것도 고려해야함.)
        for origin_meta in list(TEST_DOCUMENT.metadata.items()):
            assert origin_meta in list(passages[passages_num].metadata_etc.items())

        # Check header value store into metadata_etc properly
        for key, value in header_info[passages_num].items():
            assert (key, value) in list(passages[passages_num].metadata_etc.items())
