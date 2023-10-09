from data_preparation.create_dataset import (
    _to_valid_patch_dimension, RectangleDim
)


# Write 4 tests for _to_valid_patch_dimension
def test_to_valid_patch_dimension():
    # Valid rectangles
    rec = RectangleDim(0, 240, 0, 80)
    assert _to_valid_patch_dimension(rec) == rec

    rec = RectangleDim(0, 210, 0, 70)
    assert _to_valid_patch_dimension(rec) == rec

    # Smaller than min width
    rec = RectangleDim(100, 220, 30, 70)
    assert _to_valid_patch_dimension(rec) == RectangleDim(55, 265, 15, 85)

    # Smaller ratio
    rec = RectangleDim(0, 240, 30, 70)
    assert _to_valid_patch_dimension(rec) == RectangleDim(0, 240, 10, 90)

    # Larger ratio
    rec = RectangleDim(0, 240, 0, 100)
    assert _to_valid_patch_dimension(rec) == RectangleDim(0, 240, 10, 90)
