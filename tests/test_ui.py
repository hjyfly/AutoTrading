import py_compile


def test_화면_파일이_문법_오류_없이_컴파일된다():
    """화면(src/ui.py)은 다른 테스트가 불러오지 않으므로 문법만 따로 확인한다."""
    py_compile.compile("src/ui.py", doraise=True)
