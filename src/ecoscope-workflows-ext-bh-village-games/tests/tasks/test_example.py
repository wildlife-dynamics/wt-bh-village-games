from ecoscope_workflows_ext_bh_village_games.tasks import (
    add_one_thousand,
)


def test_add_one_thousand():
    assert 1001 == add_one_thousand(1)
