from test_data import TestDataContainer
import numpy as np


if __name__ == '__main__':
    errors = [90 / 320 * i for i in [-0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]]
    bound_in_px = 8
    cyclo_bound_in_deg = 5
    distance = 2

    test_conf = TestDataContainer(
        stimulus=range(20),
        object_distance=[distance],
        pan_error=errors,
        tilt_error=errors,
        vergence_error=errors,
        cyclo_pos=np.linspace(-cyclo_bound_in_deg, cyclo_bound_in_deg, 8).astype(np.float32),
        n_iterations=20,
        name="wrt_vergence_error_only_at_2m"
    )
    test_conf.add_wrt_vergence_error(bound_in_px)
    test_conf.dump("../config/test_conf/")

    test_conf = TestDataContainer(
        stimulus=range(20),
        object_distance=[distance],
        pan_error=errors,
        tilt_error=errors,
        vergence_error=errors,
        cyclo_pos=np.linspace(-cyclo_bound_in_deg, cyclo_bound_in_deg, 8).astype(np.float32),
        n_iterations=20,
        name="wrt_cyclo_error_only_at_2m"
    )
    test_conf.add_wrt_cyclo_pos(cyclo_bound_in_deg)
    test_conf.dump("../config/test_conf/")

    test_conf = TestDataContainer(
        stimulus=range(20),
        object_distance=[distance],
        pan_error=errors,
        tilt_error=errors,
        vergence_error=errors,
        cyclo_pos=np.linspace(-cyclo_bound_in_deg, cyclo_bound_in_deg, 8).astype(np.float32),
        n_iterations=20,
        name="wrt_pan_tilt_error_only_at_2m"
    )
    test_conf.add_wrt_speed_error("pan", bound_in_px)
    test_conf.add_wrt_speed_error("tilt", bound_in_px)
    test_conf.dump("../config/test_conf/")

    test_conf = TestDataContainer(
        stimulus=range(20),
        object_distance=[distance],
        pan_error=errors,
        tilt_error=errors,
        vergence_error=errors,
        cyclo_pos=np.linspace(-cyclo_bound_in_deg, cyclo_bound_in_deg, 8).astype(np.float32),
        n_iterations=20,
        name="wrt_tilt_error_only_at_2m"
    )
    test_conf.dump("../config/test_conf/")
