import pytest
import infrence_engine.error as er

# #############################################################################
# #############################################################################
# raise_type_error
# #############################################################################
# #############################################################################


def test_raise_type_error():
    with pytest.raises(TypeError) as err:
        er.raise_type_error("1", "Bob", ["int"])
    assert "Bob should be type int, type str was passed" in str(err.value)
