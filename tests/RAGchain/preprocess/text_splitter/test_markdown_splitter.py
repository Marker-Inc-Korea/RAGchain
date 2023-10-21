import copy

import pytest
from langchain.schema import Document

from RAGchain.preprocess.text_splitter import MarkDownHeaderSplitter

TEST_DOCUMENT = Document(
    page_content="""
    # 무야호 할아버지
    
    무야호 할아버지는 대한민국의 인터넷 유머 캐릭터로, 그의 특유의 말투와 행동이 네티즌들 사이에 인기를 끌었습니다. 
    
    ### 생애
    
    무야호 할아버지의 본명은 김영관이며, 충청남도 천안시 동남구에 거주하고 있습니다. 그는 1960년대에 전국노래자랑에 출연하여 우승한 경력이 있는 가수이며, 그의 노래 실력은 여전히 탄탄합니다.
    
    ## 무야호
    
    "무야호"라는 말은 그의 대표적인 말투로, 이는 일본 애니메이션 '원피스'의 주인공 루피가 자주 사용하는 말에서 유래되었습니다. 그는 이 말을 자주 사용하여 자신의 기분을 표현합니다.
    
    ### 욱까
    욱까 새끼들~ 부들부들 하구나~ 아아 즐겁구나 주말이~
    
    ## 리랭크
    리랭크도 나도 몰라 동건이도~
    
    # 리중딱

    안하긴뭘안해~~ 반갑습니다~~ 이피엘에서 우승못하는팀 누구야? 소리질러~~!!  
    리중딱 리중딱 신나는노래~ 나도한번 불러본다~~(박수) (박수) (박수) 짠리잔짠~~  
    우리는 우승하기 싫~어~ 왜냐면 우승하기 싫은팀이니깐~ 20년 내~내~ 프리미어리그~ 우승도 못하는 우리팀이다.  
    리중딱 리중딱 신나는노래 ~~~ 나도한번불러본다~  
    리중딱 리중딱 신나는노래 ~~ 가슴치며 불러본다~  
    리중딱 노래가사는~ 생활과 정보가 있는노래 중딱이~~와 함께라면 제~라드도함께 우승못한다.
    
    ### 맨까송
    맨까 새끼들 부들부들하구나
    아아~ 즐겁구나 주 말 이~

    """,
    metadata={
        'source': 'test_source',
        # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
        '감스트 나락감지': '열심히 하시잖아',
        '감스트 중노': '그쪽도 맹박사님을 아세요?',
        'Data information': 'test for markdownheader splitter',
        '좋았어': 'lets go'
    }
)


@pytest.fixture
def markdownheader_text_splitter():
    markdownheader_text_splitter = MarkDownHeaderSplitter()
    yield markdownheader_text_splitter


def test_markdownheader_text_splitter(markdownheader_text_splitter):
    passages = markdownheader_text_splitter.split_document(TEST_DOCUMENT)

    assert len(passages) > 1
    assert passages[0].next_passage_id == passages[1].id
    assert passages[1].previous_passage_id == passages[0].id
    assert passages[0].filepath == 'test_source'
    assert passages[0].filepath == passages[1].filepath
    assert passages[0].previous_passage_id is None
    assert passages[-1].next_passage_id is None


    # Check splitter preserve other metadata in original document.
    ## Remove file path information in clone of TEST_DOCUMENT for test.
    test_document_for_test = copy.deepcopy(TEST_DOCUMENT)
    test_document_for_test.metadata.pop('source')

    for passage in passages:
        for origin_meta in list(test_document_for_test.metadata.items()):
            assert origin_meta in list(passage.metadata_etc.items())

    # Check Markdown information put in metadata_etc right form.
    ## Front part of Test document
    assert ('Header 1', '무야호 할아버지') in list(passages[0].metadata_etc.items())

    ## Middle part of Test document(To verify if the parent Header 2 '무야호' has been changed to parent Header 2. '리랭크')
    assert ('Header 1', '무야호 할아버지') in list(passages[0].metadata_etc.items()) and ('Header 2', '리랭크') in list(
        passages[4].metadata_etc.items())

    ## End part of Test document
    assert ('Header 1', '리중딱') in list(passages[-1].metadata_etc.items()) and ('Header 3', '맨까송') in list(
        passages[-1].metadata_etc.items())
