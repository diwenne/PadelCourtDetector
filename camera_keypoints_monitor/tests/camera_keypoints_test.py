from src.camera_keypoints_main import almost_equal


def test_camera_keypoints():
    # Roughly 6 pixel move would trigger
    width = 100
    height = 100
    threshold_of_diagonal = 0.05

    # Should trigger
    x = (10, 15)
    y = (5, 10)

    assert not almost_equal(x, y, width, height, threshold_of_diagonal)

    # Should not trigger
    x = (10, 15)
    y = (7, 18)

    assert almost_equal(x, y, width, height, threshold_of_diagonal)
    print("Test test_camera_keypoints passed")


if __name__ == "__main__":
    test_camera_keypoints()