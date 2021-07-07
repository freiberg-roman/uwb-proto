import numpy as np

from uwb.generator import BlobGenerator


def test_generation_data():
    bg = BlobGenerator(
        grid_dims=[2, 4, 6],
        step_size=10,
        measurements_per_location=25,
        modal_range=(1, 1),
        deviation=1.0,
    )

    # assert that write grid is allocated
    assert len(bg.grid) == 3
    assert len(bg.grid[0]) == 2
    assert len(bg.grid[1]) == 4
    assert len(bg.grid[2]) == 6
    assert bg.shape == (2, 4, 6)

    # generation of data
    assert bg._data is None
    samples = bg.gen()
    assert bg._data is not None
    assert len(samples[0, 0, 0, :]) == 25  # 25 measurements per location
    assert samples.dtype == np.float64


def test_data_iteration():
    bg = BlobGenerator(
        grid_dims=[2, 4, 6],
        step_size=10,
        measurements_per_location=100,
        modal_range=(1, 5),
        deviation=1.0,
    )

    for i, (data, idx, position) in enumerate(bg):
        if i == 0:  # first entry
            assert idx == (0, 0, 0)
            assert data.shape == (100, 3)  # 100 three dimensional measurements
            assert position[0] == 10
            assert position[1] == 10
            assert position[2] == 10

        if i == 2 * 4 * 6:  # last entry
            assert idx == (1, 3, 5)
            assert data.shape == (100, 3)  # 100 three dimensional measurements
            assert tuple(map(tuple, np.asarray(position))) == (20, 40, 60)


def test_closest_position_in_grid():
    bg = BlobGenerator(
        grid_dims=[2, 4, 6],
        step_size=10,
        measurements_per_location=100,
        modal_range=(1, 5),
        deviation=1.0,
    )
    bg.gen()

    # coordinates that do undershoot, overshoot the grid and one is on  the map.
    to_find_locations = np.array(
        [[0.0, 0.0, 1.0], [100.0, 100.0, 100.0], [11.0, 16.0, 30.0]]
    )

    pos, _ = bg.get_closest_position(to_find_locations)
    assert tuple(np.asarray(pos[0])) == (0, 0, 0)  # position (10, 10, 10)
    assert tuple(np.asarray(pos[1])) == (1, 3, 5)  # position (20, 40, 60)
    assert tuple(np.asarray(pos[2])) == (0, 1, 2)  # position (10, 20, 30)
