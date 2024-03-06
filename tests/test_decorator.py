from flextrain.diffusion.utils import catch_all_and_log


@catch_all_and_log
def fn_1(value):
    return value + 1


@catch_all_and_log
def fn_2(value):
    raise RuntimeError('Hahah, exception raised!')


def test_catch_all_and_log_without_exception():
    r = fn_1(42)
    assert r == 43


def test_catch_all_and_log_wit_exception():
    r = fn_2(42)
    assert r is None
