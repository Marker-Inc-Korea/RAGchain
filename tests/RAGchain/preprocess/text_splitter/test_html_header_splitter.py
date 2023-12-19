import copy

import pytest
from langchain.schema import Document

from RAGchain.preprocess.text_splitter import HTMLHeaderSplitter

TEST_DOCUMENT = Document(
    page_content="""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>노동요</title>
    </head>
    <body>
        <div>
            <h1>학박사님을 아세유?</h1>
                <p>안하긴뭘 안해~ 월요일부터 일찍 일어나는 사람 누구야~ 소리질러!</p>
                <h2>학교가는 동규형</h2>
                <div>
                    <p>
                    학 학 학 학 학 학<br>
                    학박사님을 아세요??<br>
                    학 학 학 학 학 학<br>
                    </p>
                </div>
                    <h3> 근데 리뷰할때 동규형이 보면 어떡하지</h3>
        </div>

        <div>
            <h1>리중딱</h1>
                <h2>감스트</h2>
                <div>
                <p>
                안하긴뭘안해~~ 반갑습니다~~ 이피엘에서 우승못하는팀 누구야? 소리질러~~!!<br>
                리중딱 리중딱 신나는노래~ 나도한번 불러본다~~(박수) (박수) (박수) 짠리잔짠~~<br>
                우리는 우승하기 싫~어~ 왜냐면 우승하기 싫은팀이니깐~ 20년 내~내~ 프리미어리그~ 우승도 못하는 우리팀이다.<br>
                리중딱 리중딱 신나는노래 ~~~ 나도한번불러본다~<br>
                리중딱 리중딱 신나는노래 ~~ 가슴치며 불러본다~<br>
                리중딱 노래가사는~ 생활과 정보가 있는노래 중딱이~~와 함께라면 제~라드도함께 우승못한다.
                </p>
                </div>
            <h3>근데 ragchain 쓰는 사람이 리버풀팬이면 어떡하지</h3>
            <div>
            <p>
            난 몰라유 그딴거 잘 몰라유
            </p>
            </div>
        </div> 

        <div>
            <h1>맨까송</h1>
                <h2>감빡이</h2>
                <div>
                <p>
                맨까 새끼들 부들부들하구나<br>
                억까를 해 봐도 우린 골 넣지<br>
                니네가 아무리 맹구다 어쩐다고 놀려도<br>
                아아~ 즐겁구나 명 절 이~(짜스!)<br>
                맨까 새끼들 부들부들하구나<br>
                살짝쿵 설렜니 아니 안 되지<br>
                이겨도 지롤 져도 지롤 뭐만 하면 리그컵<br>
                아~ 리그컵도 축 군 데~ (컴온!!)<br>
                맨까 새끼들 부들부들하구나<br>
                돌아온 미친 폼 누가 막을래?<br>
                더 보기 리그 탈출 직전[다른가사2] 돌아와요 맨유 팬!<br>
                아~ 기대된다 챔 스 가~ Siuuuuuuu!<br>
                </p>
                </div>
            <h3>근데 ragchain 쓰는 사람이 맨유팬이면 어떡하지</h3>
                <div>
                    <p>
                        열심히 하시잖아~, 그만큼 열심히 하신다는거지~
                    </p>
                </div>
        </div>    

        </div>
    </body>
    </html>
    """,
    metadata={
        'source': 'test_source',
        # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
        'Data information': 'test for htmldownheader splitter',
        '근본과 실력 둘다 있는 팀': '레알마드리드',
        '근본만 충만한 팀': '리버풀',
        '실력은 있으나 노근본인팀': '파리 생제르망',
        '둘다 없는 팀': '토트넘 홋스퍼'
    }
)


@pytest.fixture
def html_header_text_splitter():
    html_header_text_splitter = HTMLHeaderSplitter()
    yield html_header_text_splitter


def test_html_header_text_splitter(html_header_text_splitter):
    passages = html_header_text_splitter.split_document(TEST_DOCUMENT)

    assert len(passages) > 1
    assert passages[0].next_passage_id == passages[1].id
    assert passages[1].previous_passage_id == passages[0].id
    assert passages[0].filepath == 'test_source'
    assert passages[0].filepath == passages[1].filepath
    assert passages[0].previous_passage_id is None
    assert passages[-1].next_passage_id is None

    # Check first passage whether it contains header information of fist layout(first div).
    assert ('학박사님을 아세유? 학교가는 동규형 근데 리뷰할때 동규형이 보면 어떡하지') in passages[0].content

    # Check splitter preserve other metadata in original document.
    test_document_metadata = list(copy.deepcopy(TEST_DOCUMENT).metadata.items())
    test_document_metadata.pop(0)
    for element in test_document_metadata:
        assert element in list(passages[1].metadata_etc.items())

    # Check passages' metadata_etc
    ## metadata_etc can't contain file path(Except first part of first div).
    assert ('source', 'test_source') not in list(passages[1].metadata_etc.items())
    assert ('source', 'test_source') not in list(passages[-1].metadata_etc.items())

    # Check HTML header information put in metadata_etc right form.
    assert ('Header 1', '학박사님을 아세유?') in list(passages[1].metadata_etc.items())

    assert ('Header 1', '맨까송') in list(passages[-1].metadata_etc.items())
    assert ('Header 2', '감빡이') in list(passages[-1].metadata_etc.items())
    assert ('Header 3', '근데 ragchain 쓰는 사람이 맨유팬이면 어떡하지') in list(passages[-1].metadata_etc.items())
