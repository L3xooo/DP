from td3.utils.prev_curr import PrevCurr


def test_init_sets_prev_and_curr():
    pc = PrevCurr(prev=1, curr=2)
    assert pc.prev == 1
    assert pc.curr == 2


def test_set_prev():
    pc = PrevCurr(prev=0, curr=0)
    pc.set_prev(10)
    assert pc.prev == 10
    assert pc.curr == 0


def test_set_curr():
    pc = PrevCurr(prev=0, curr=0)
    pc.set_curr(20)
    assert pc.prev == 0
    assert pc.curr == 20


def test_set_both():
    pc = PrevCurr(prev=1, curr=2)
    pc.set_both(5)
    assert pc.prev == 5
    assert pc.curr == 5


def test_set_prev_from_curr():
    pc = PrevCurr(prev=1, curr=99)
    pc.set_prev_from_curr()
    assert pc.prev == 99
    assert pc.curr == 99


def test_update_moves_curr_to_prev_and_sets_new_curr():
    pc = PrevCurr(prev="a", curr="b")
    pc.update("c")
    assert pc.prev == "b"
    assert pc.curr == "c"