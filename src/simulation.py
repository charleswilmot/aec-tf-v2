from pyrep import PyRep
from pyrep.objects import VisionSensor
from pyrep.const import RenderMode
import multiprocessing as mp
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from skimage.transform import resize
from contextlib import contextmanager
from traceback import format_exc
from custom_shapes import Head, Screen, UniformMotionScreen
import time


MODEL_PATH = os.environ["COPPELIASIM_MODEL_PATH"]
Y_EYES_DISTANCE = 0.034


def distance_to_vergence(distance):
    return - np.rad2deg(2 * np.arctan2(Y_EYES_DISTANCE, distance))

class SimulationConsumerFailed(Exception):
    def __init__(self, consumer_exception, consumer_traceback):
        self.consumer_exception = consumer_exception
        self.consumer_traceback = consumer_traceback

    def __str__(self):
        return '\n\nFROM CONSUMER:\n\n{}'.format(self.consumer_traceback)

def communicate_return_value(method):
    """method from the SimulationConsumer class decorated with this function
    will send there return value to the SimulationProducer class"""
    method._communicate_return_value = True
    return method


def default_dont_communicate_return(cls):
    """Class decorator for the SimulationConsumers meaning that by default, all
    methods don't communicate their return value to the Producer class"""
    for attribute_name, attribute in cls.__dict__.items():
        if callable(attribute):
            communicate = hasattr(attribute, '_communicate_return_value')
            attribute._communicate_return_value = communicate
    return cls


def c2p_convertion_function(cls, method):
    """Function that transform a Consumer method into a Producer method.
    It add a blocking flag that determines whether the call is blocking or not.
    If you call a `Producer.mothod(blocking=False)`, you then must
    `Producer._wait_for_answer()`"""
    def new_method(self, *args, blocking=True, **kwargs):
        cls._send_command(self, method, *args, **kwargs)
        if method._communicate_return_value and blocking:
            return cls._wait_for_answer(self)
    new_method._communicate_return_value = method._communicate_return_value
    return new_method


def consumer_to_producer_method_conversion(cls):
    """Class decorator that transforms all methods from the Consumer to the
    Producer, except for methods starting with an '_', and for the
    multiprocessing.Process methods"""
    proc_methods = [
        "run", "is_alive", "join", "kill", "start", "terminate", "close"
    ]
    method_dict = {
        **SimulationConsumerAbstract.__dict__,
        **SimulationConsumer.__dict__,
    }
    convertables = {
        method_name: method \
        for method_name, method in method_dict.items()\
        if callable(method) and\
        method_name not in proc_methods and\
        not method_name.startswith("_")
    }
    for method_name, method in convertables.items():
        new_method = c2p_convertion_function(cls, method)
        setattr(cls, method_name, new_method)
    return cls


def p2p_convertion_function(name):
    """This function transforms a producer method into a Pool method"""
    def new_method(self, *args, **kwargs):
        if self._distribute_args_mode:
            # all args are iterables that must be distributed to each producer
            for i, producer in enumerate(self._active_producers):
                getattr(producer, name)(
                    *[arg[i] for arg in args],
                    blocking=False,
                    **{key: value[i] for key, value in kwargs.items()}
                )
        else:
            for producer in self._active_producers:
                getattr(producer, name)(*args, blocking=False, **kwargs)
        if getattr(SimulationProducer, name)._communicate_return_value:
            return [
                producer._wait_for_answer() for producer in self._active_producers
            ]
    return new_method

def producer_to_pool_method_convertion(cls):
    """This class decorator transforms all Producer methods (besides close and
    methods starting with '_') to the Pool object."""
    convertables = {
        method_name: method \
        for method_name, method in SimulationProducer.__dict__.items()\
        if callable(method) and not method_name.startswith("_")\
        and not method_name == 'close'
    }
    for method_name, method in convertables.items():
        new_method = p2p_convertion_function(method_name)
        setattr(cls, method_name, new_method)
    return cls


@default_dont_communicate_return
class SimulationConsumerAbstract(mp.Process):
    _id = 0
    """This class sole purpose is to better 'hide' all interprocess related code
    from the user."""
    def __init__(self, process_io, scene=MODEL_PATH + "/empty_scene.ttt", gui=False):
        super().__init__(
            name="simulation_consumer_{}".format(SimulationConsumerAbstract._id)
        )
        self._id = SimulationConsumerAbstract._id
        SimulationConsumerAbstract._id += 1
        self._scene = scene
        self._gui = gui
        self._process_io = process_io
        np.random.seed()

    def run(self):
        self._pyrep = PyRep()
        self._pyrep.launch(
            self._scene,
            headless=not self._gui,
            write_coppeliasim_stdout_to_file=False
        )
        self._process_io["simulaton_ready"].set()
        self._main_loop()

    def _close_pipes(self):
        self._process_io["command_pipe_out"].close()
        self._process_io["return_value_pipe_in"].close()
        # self._process_io["exception_pipe_in"].close() # let this one open

    def _main_loop(self):
        success = True
        while success and not self._process_io["must_quit"].is_set():
            success = self._consume_command()
        self._pyrep.shutdown()
        self._close_pipes()

    def _consume_command(self):
        try: # to execute the command and send result
            success = True
            command = self._process_io["command_pipe_out"].recv()
            self._process_io["slot_in_command_queue"].release()
            ret = command[0](self, *command[1], **command[2])
            if command[0]._communicate_return_value:
                self._communicate_return_value(ret)
        except Exception as e: # print traceback, dont raise
            traceback = format_exc()
            success = False # return False: quit the main loop
            self._process_io["exception_pipe_in"].send((e, traceback))
        finally:
            return success

    def _communicate_return_value(self, value):
        self._process_io["return_value_pipe_in"].send(value)

    def signal_command_pipe_empty(self):
        self._process_io["command_pipe_empty"].set()
        time.sleep(0.1)
        self._process_io["command_pipe_empty"].clear()

    def good_bye(self):
        pass

    @communicate_return_value
    def blocking_barrier(self):
        return None


@default_dont_communicate_return
class SimulationConsumer(SimulationConsumerAbstract):
    def __init__(self, process_io, scene=MODEL_PATH + "/empty_scene.ttt", gui=False):
        super().__init__(process_io, scene, gui)
        self._shapes = defaultdict(list)
        self._stateful_shape_list = []
        self._state_buffer = None
        self._cams = {}
        self._screens = {}
        self._textures = {}
        self.head = None
        self.background = None
        self.uniform_motion_screen = None
        self.scales = {}
        self.fake_scales = {}

    def add_head(self):
        if self.head is None:
            model = self._pyrep.import_model(MODEL_PATH + "/head.ttm")
            model = Head(model.get_handle())
            self.head = model
            return self.head
        else:
            raise ValueError("Can not add two heads to the simulation at the same time")

    def add_background(self, name):
        if self.background is None:
            model = self._pyrep.import_model(MODEL_PATH + "/{}.ttm".format(name))
            self.background = model
            return self.background
        else:
            raise ValueError("Can not add two backgrounds to the simulation at the same time")

    def add_textures(self, textures_path):
        textures_names = os.listdir(textures_path)
        textures_list = []
        for name in textures_names:
            if name not in self._textures:
                self._textures[name] = self._pyrep.create_texture(
                    os.path.normpath(textures_path + '/' + name))[1]
            textures_list.append(self._textures[name])
        return textures_list

    def add_screen(self, textures_path, size=1.5):
        textures_list = self.add_textures(textures_path)
        screen = Screen(textures_list, size=size)
        screen_id = screen.get_handle()
        self._screens[screen_id] = screen
        return screen_id

    def add_uniform_motion_screen(self, textures_path, size=1.5,
            min_distance=0.5, max_distance=5.0,
            max_depth_speed=0.03, max_speed_in_deg=1.125):
        if self.uniform_motion_screen is not None:
            raise ValueError("Can not add multiple uniform motion screens")
        else:
            textures_list = self.add_textures(textures_path)
            screen = UniformMotionScreen(textures_list, size, min_distance,
                max_distance, max_depth_speed, max_speed_in_deg)
            screen_id = screen.get_handle()
            self._screens[screen_id] = screen
            self.uniform_motion_screen = screen
            return screen_id

    def episode_reset_uniform_motion_screen(self, start_distance=None,
            depth_speed=None, angular_speed=None, direction=None,
            texture_id=None, preinit=False):
        if self.uniform_motion_screen is None:
            raise ValueError("No uniform motion screens in the simulation")
        else:
            self.uniform_motion_screen.episode_reset(start_distance,
                depth_speed, angular_speed, direction, texture_id, preinit)

    @communicate_return_value
    def get_screen_position(self):
        if self.uniform_motion_screen is None:
            raise ValueError("Can not get screen position: no screen")
        return self.uniform_motion_screen.position

    def episode_reset_head(self, vergence=None, cyclo=None):
        if self.head is None:
            raise ValueError("No head in the simulation")
        else:
            if vergence is None:
                distance = np.random.uniform(low=0.5, high=5)
                vergence = distance_to_vergence(distance)
            if cyclo is None:
                cyclo = 0
            self.head.reset(vergence=vergence, cyclo=cyclo)
            # for i in range(10):
            self._pyrep.step()

    def move_uniform_motion_screen(self):
        if self.uniform_motion_screen is None:
            raise ValueError("No uniform motion screens in the simulation")
        else:
            self.uniform_motion_screen.move()

    def add_camera(self, eye, resolution, view_angle):
        if self.head is None:
            raise ValueError("Can not add a camera with no head")
        else:
            position = self.head.get_eye_position(eye)
            orientation = self.head.get_eye_orientation(eye)
            vision_sensor = VisionSensor.create(
                resolution=resolution,
                position=position,
                orientation=orientation,
                view_angle=view_angle,
                far_clipping_plane=100.0,
                render_mode=RenderMode.OPENGL,
            )
            vision_sensor.set_parent(self.head.get_eye_parent(eye))
            cam_id = vision_sensor.get_handle()
            self._cams[cam_id] = vision_sensor
            return cam_id

    def delete_camera(self, cam_id):
        self._cams[cam_id].remove()
        self._cams.pop(cam_id)

    @communicate_return_value
    def get_vision(self, color_scaling=None):
        if color_scaling is None:
            return {
                scale_id: np.concatenate([
                    self._cams[left].capture_rgb(),
                    self._cams[right].capture_rgb()
                    ], axis=-1) * 2 - 1
                for scale_id, (left, right) in self.scales.items()
            }
        else:
            return {
                scale_id: np.concatenate([
                    self._cams[left].capture_rgb(),
                    self._cams[right].capture_rgb()
                    ], axis=-1) * color_scaling[scale_id] * 2 - 1
                for scale_id, (left, right) in self.scales.items()
            }

    @communicate_return_value
    def get_fake_vision(self, color_scaling=None):
        vision = self.get_vision(color_scaling=color_scaling)
        return {
            scale_id: resize(
                vision["fake_scale"][scale_desc.start:scale_desc.stop, scale_desc.start:scale_desc.stop],
                scale_desc.resolution, anti_aliasing=True)
            for scale_id, scale_desc in self.fake_scales.items()
        }

    @communicate_return_value
    def add_scale(self, id, resolution, view_angle):
        if id in self.scales:
            raise ValueError("Scale with id {} is already present".format(id))
        else:
            self.head.set_joints_velocities(0, 0, 0, 0)
            self.head.set_joints_positions(0, 0, 0, 0)
            # self._pyrep.step() # step to make sure that the eyes reached the target position of 0 (might be useless)
            left = self.add_camera('left', resolution, view_angle)
            right = self.add_camera('right', resolution, view_angle)
            self.scales[id] = (left, right)
            return id

    @communicate_return_value
    def get_scales(self):
        return list(self.scales.keys())

    def delete_scale(self, id):
        if id in self.scales:
            left, right = self.scales[id]
            self.delete_camera(left)
            self.delete_camera(right)
            self.scales.pop(id)
        else:
            raise ValueError("Scale with id {} does not exist".format(id))

    @communicate_return_value
    def add_fake_scales(self, scales):
        for s in scales.values():
            if s.resolution[0] != s.resolution[1]:
                raise ValueError("The fake scales must be square!")
        max_view_angle = max(s.view_angle for s in scales.values())
        min_pixel_size = min(s.view_angle / s.resolution[0] for s in scales.values())
        max_resolution = int(np.ceil(max_view_angle / min_pixel_size))
        resolution = [max_resolution, max_resolution]
        self.add_scale("fake_scale", resolution, max_view_angle)
        print(resolution, max_view_angle)
        self.fake_scales.update(scales)
        for scale, scale_desc in scales.items():
            downscaling_factor = scale_desc.view_angle / scale_desc.resolution[0] / min_pixel_size
            n_pixels = scale_desc.view_angle * max_resolution / max_view_angle
            start = int((max_resolution - n_pixels) / 2)
            stop = int(start + scale_desc.resolution[0] * downscaling_factor)
            self.fake_scales[scale].start = start
            self.fake_scales[scale].stop = stop
            print(scale, start, stop)

    def apply_action(self, action):
        tilt_acceleration, pan_acceleration, vergence_velocity, cyclo_velocity = action
        self.head.set_action(tilt_acceleration, pan_acceleration, vergence_velocity, cyclo_velocity)
        self.step_sim()

    @communicate_return_value
    def get_joints_errors(self):
        screen_distance = self.uniform_motion_screen.distance
        screen_vergence = distance_to_vergence(screen_distance)
        screen_tilt_speed, screen_pan_speed = self.uniform_motion_screen.tilt_pan_speed
        _, _, eyes_vergence, _ = self.head.get_joints_positions()
        eyes_tilt_speed, eyes_pan_speed, _, _ = self.head.get_joints_velocities()
        vergence_error = eyes_vergence - screen_vergence
        tilt_error = eyes_tilt_speed - screen_tilt_speed
        pan_error = eyes_pan_speed - screen_pan_speed
        return tilt_error, pan_error, vergence_error

    @communicate_return_value
    def get_joints_velocities(self):
        return self.head.get_joints_velocities()

    @communicate_return_value
    def get_joints_positions(self):
        return self.head.get_joints_positions()

    def set_control_loop_enabled(self, bool):
        self.head.set_control_loop_enabled(bool)

    def set_motor_enabled(self, bool):
        self.head.set_motor_enabled(bool)

    def set_motor_locked_at_zero_velocity(self, bool):
        self.head.set_motor_locked_at_zero_velocity(bool)

    def step_sim(self):
        if self.uniform_motion_screen is not None:
            self.uniform_motion_screen.move()
        self._pyrep.step()

    def start_sim(self):
        self._pyrep.start()

    def stop_sim(self):
        self._pyrep.stop()

    @communicate_return_value
    def get_simulation_timestep(self):
        return self._pyrep.get_simulation_timestep()


@consumer_to_producer_method_conversion
class SimulationProducer(object):
    def __init__(self, scene=MODEL_PATH + "/empty_scene.ttt", gui=False):
        self._process_io = {}
        self._process_io["must_quit"] = mp.Event()
        self._process_io["simulaton_ready"] = mp.Event()
        self._process_io["command_pipe_empty"] = mp.Event()
        self._process_io["slot_in_command_queue"] = mp.Semaphore(100)
        pipe_out, pipe_in = mp.Pipe(duplex=False)
        self._process_io["command_pipe_in"] = pipe_in
        self._process_io["command_pipe_out"] = pipe_out
        pipe_out, pipe_in = mp.Pipe(duplex=False)
        self._process_io["return_value_pipe_in"] = pipe_in
        self._process_io["return_value_pipe_out"] = pipe_out
        pipe_out, pipe_in = mp.Pipe(duplex=False)
        self._process_io["exception_pipe_in"] = pipe_in
        self._process_io["exception_pipe_out"] = pipe_out
        self._consumer = SimulationConsumer(self._process_io, scene, gui=gui)
        self._consumer.start()
        print("consumer {} started".format(self._consumer._id))
        self._closed = False
        # atexit.register(self.close)

    def _get_process_io(self):
        return self._process_io

    def _check_consumer_alive(self):
        if not self._consumer.is_alive():
            self._consumer.join()
            print("### My friend ({}) died ;( raising its exception: ###\n".format(self._consumer._id))
            self._consumer.join()
            self._closed = True
            exc, traceback = self._process_io["exception_pipe_out"].recv()
            raise SimulationConsumerFailed(exc, traceback)
        return True

    def _send_command(self, function, *args, **kwargs):
        self._process_io["command_pipe_in"].send((function, args, kwargs))
        semaphore = self._process_io["slot_in_command_queue"]
        while not semaphore.acquire(block=False, timeout=0.1):
            self._check_consumer_alive()

    def _wait_for_answer(self):
        while not self._process_io["return_value_pipe_out"].poll(1):
            # print(method, "waiting for an answer...nothing yet...alive?")
            self._check_consumer_alive()
        answer = self._process_io["return_value_pipe_out"].recv()
        # print(method, "waiting for an answer...got it!")
        return answer

    def _wait_consumer_ready(self):
        self._process_io["simulaton_ready"].wait()

    def close(self):
        if not self._closed:
            # print("Producer closing")
            if self._consumer.is_alive():
                self._wait_command_pipe_empty()
                # print("command pipe empty, setting must_quit flag")
                self._process_io["must_quit"].set()
                # print("flushing command pipe")
                self.good_bye()
            self._closed = True
            # print("succesfully closed")
            self._consumer.join()
            print("consumer {} closed".format(self._consumer._id))
        else:
            print("{} already closed, doing nothing".format(self._consumer._id))

    def _wait_command_pipe_empty(self):
        self._send_command(SimulationConsumer.signal_command_pipe_empty)
        self._process_io["command_pipe_empty"].wait()

    def __del__(self):
        self.close()


@producer_to_pool_method_convertion
class SimulationPool:
    def __init__(self, size, scene=MODEL_PATH + "/empty_scene.ttt", guis=[]):
        self._producers = [
            SimulationProducer(scene, gui=i in guis) for i in range(size)
        ]
        self.n = size
        self._active_producers_indices = list(range(size))
        self._distribute_args_mode = False
        self.wait_consumer_ready()

    @contextmanager
    def specific(self, list_or_int):
        _active_producers_indices_before = self._active_producers_indices
        indices = list_or_int if type(list_or_int) is list else [list_or_int]
        self._active_producers_indices = indices
        yield
        self._active_producers_indices = _active_producers_indices_before

    @contextmanager
    def distribute_args(self):
        self._distribute_args_mode = True
        yield
        self._distribute_args_mode = False

    def _get_active_producers(self):
        return [self._producers[i] for i in self._active_producers_indices]
    _active_producers = property(_get_active_producers)

    def close(self):
        for producer in self._producers:
            producer.close()

    def wait_consumer_ready(self):
        for producer in self._producers:
            producer._wait_consumer_ready()



if __name__ == '__main__':
    def test_1():
        simulation = SimulationProducer(gui=False)
        simulation.start_sim()
        simulation.step_sim()
        simulation.add_background("ny_times_square")
        simulation.add_head()
        simulation.add_scale("fine", (32, 32), 2.0)
        simulation.add_scale("coarse", (32, 32), 6.0)
        simulation.step_sim()
        N = 1000
        t0 = time.time()
        for i in range(N):
            vision = simulation.get_vision()
            simulation.step_sim()
        t1 = time.time()

        print("\n")
        print(vision)
        print("\n")
        print("{:.3f} FPS".format(N / (t1 - t0)))
        simulation.stop_sim()

    def test_2():
        simulation = SimulationProducer(gui=True)
        simulation.start_sim()
        simulation.step_sim()
        simulation.add_head()
        simulation.add_uniform_motion_screen("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/", size=1.5)
        for i in range(100):
            simulation.episode_reset_uniform_motion_screen()
            for j in range(20):
                print(i, end='\r')
                simulation.move_uniform_motion_screen()
                simulation.step_sim()
        simulation.stop_sim()

    def test_3():
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure()
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        simulation = SimulationProducer(gui=True)
        simulation.add_background("ny_times_square")
        simulation.add_head()
        simulation.add_scale("fine", (32, 32), 9.0)
        simulation.add_scale("coarse", (320, 320), 27.0)
        simulation.add_uniform_motion_screen("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/", size=1.5)
        simulation.start_sim()
        simulation.step_sim()
        vision = simulation.get_vision() # block
        im0 = ax0.imshow((127.5 + 127.5 * vision["coarse"][..., :3]).astype(np.uint8))
        im1 = ax1.imshow((127.5 + 127.5 * vision["coarse"][..., 3:]).astype(np.uint8))
        plt.show()
        for i in range(10):
            print(i)
            simulation.episode_reset_uniform_motion_screen()
            simulation.episode_reset_head()
            for j in range(10):
                vision = simulation.get_vision() # block
                im0.set_data((127.5 + 127.5 * vision["coarse"][..., :3]).astype(np.uint8))
                im1.set_data((127.5 + 127.5 * vision["coarse"][..., 3:]).astype(np.uint8))
                fig.canvas.flush_events()
                simulation.apply_action(np.random.uniform(low=-1, high=1, size=4) * np.array([1.125, 1.125, 1.125, 0.125]))
                print(simulation.get_joints_positions(), simulation.get_joints_velocities())
        simulation.get_vision() # block
        simulation.stop_sim()
        plt.close(fig)

    def test_4():
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure()
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        simulation = SimulationProducer(gui=True)
        simulation.add_background("ny_times_square")
        simulation.add_head()
        simulation.add_scale("coarse", (320, 320), 27.0)
        simulation.add_uniform_motion_screen("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/", size=1.5)
        simulation.start_sim()
        simulation.step_sim()
        vision = simulation.get_vision() # block
        im0 = ax0.imshow((127.5 + 127.5 * vision["coarse"][..., :3]).astype(np.uint8))
        im1 = ax1.imshow((127.5 + 127.5 * vision["coarse"][..., 3:]).astype(np.uint8))
        plt.show()
        for i in range(10):
            print(i)
            simulation.episode_reset_uniform_motion_screen()
            simulation.episode_reset_head()
            for j in range(10):
                vision = simulation.get_vision() # block
                im0.set_data((127.5 + 127.5 * vision["coarse"][..., :3]).astype(np.uint8))
                im1.set_data((127.5 + 127.5 * vision["coarse"][..., 3:]).astype(np.uint8))
                fig.canvas.flush_events()
                simulation.apply_action([0, 0, 3, 0])
                print(simulation.get_joints_positions(), simulation.get_joints_velocities())
        simulation.get_vision() # block
        simulation.stop_sim()
        plt.close(fig)

    def test_5():
        import matplotlib.pyplot as plt
        # plt.ion()
        fig = plt.figure()
        ax00 = fig.add_subplot(221)
        ax10 = fig.add_subplot(222)
        ax01 = fig.add_subplot(223)
        ax11 = fig.add_subplot(224)
        simulations = SimulationPool(size=2) # guis=[0, 1]
        simulations.add_background("ny_times_square")
        simulations.add_head()
        simulations.add_scale("only", (320, 320), 27.0)
        simulations.add_uniform_motion_screen("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/", size=1.5)
        simulations.start_sim()
        simulations.step_sim()
        simulations.episode_reset_uniform_motion_screen(
            start_distance=2,
            depth_speed=0,
            angular_speed=0,
            direction=0,
            texture_id=0,
            preinit=False,
        )
        simulations.episode_reset_head(vergence=0, cyclo=0)
        simulations.step_sim()
        vision = simulations.get_vision() # block
        im00 = ax00.imshow((0.5 + 0.5 * vision[0]["only"][..., :3]))
        im10 = ax10.imshow((0.5 + 0.5 * vision[1]["only"][..., :3]))

        # frame01 = np.abs(vision[1]["only"][..., :3] - vision[0]["only"][..., :3])
        # frame11 = np.abs(vision[1]["only"][..., 3:] - vision[0]["only"][..., 3:])
        # frame01 /= np.max(frame01, axis=(0, 1))
        # frame11 /= np.max(frame11, axis=(0, 1))

        # frame01 = vision[1]["only"][..., :3] - vision[0]["only"][..., :3]
        # frame11 = vision[1]["only"][..., 3:] - vision[0]["only"][..., 3:]
        # frame01 -= np.min(frame01, axis=(0, 1))
        # frame11 -= np.min(frame11, axis=(0, 1))
        # frame01 /= np.max(frame01, axis=(0, 1))
        # frame11 /= np.max(frame11, axis=(0, 1))

        mean_ratio = np.mean(0.5 + 0.5 * vision[0]["only"], axis=(0, 1)) / np.mean(0.5 + 0.5 * vision[1]["only"], axis=(0, 1))
        frame = (0.5 + 0.5 * vision[1]["only"]) * mean_ratio
        frame01 = frame[..., :3]
        frame11 = frame[..., 3:]

        im01 = ax01.imshow(frame01)
        im11 = ax11.imshow(frame11)
        print(vision[0]["only"].shape)
        print(0.5 + 0.5 * np.mean(vision[0]["only"], axis=(0, 1)))
        print(0.5 + 0.5 * np.mean(vision[1]["only"], axis=(0, 1)))
        print((1 + vision[0]["only"]) / (1 + vision[1]["only"]))
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(vision[0]["only"][:, :, 0].flatten(), vision[1]["only"][:, :, 0].flatten(), color='r')
        ax.scatter(vision[0]["only"][:, :, 1].flatten(), vision[1]["only"][:, :, 1].flatten(), color='g')
        ax.scatter(vision[0]["only"][:, :, 2].flatten(), vision[1]["only"][:, :, 2].flatten(), color='b')
        plt.show()
        # for i in range(10):
        #     print(i)
        #     simulations.episode_reset_uniform_motion_screen()
        #     simulations.episode_reset_head()
        #     for j in range(10):
        #         vision = simulations.get_vision() # block
        #         im0.set_data((127.5 + 127.5 * vision["coarse"][..., :3]).astype(np.uint8))
        #         im1.set_data((127.5 + 127.5 * vision["coarse"][..., 3:]).astype(np.uint8))
        #         fig.canvas.flush_events()
        #         simulations.apply_action(np.random.uniform(low=-1, high=1, size=4) * np.array([1.125, 1.125, 1.125, 0.125]))
        #         print(simulations.get_joints_positions(), simulations.get_joints_velocities())
        # simulations.get_vision() # block
        simulations.stop_sim()
        plt.close(fig)

    def test_6():
        import matplotlib.pyplot as plt
        # plt.ion()
        fig = plt.figure()
        ax00 = fig.add_subplot(221)
        ax10 = fig.add_subplot(222)
        ax01 = fig.add_subplot(223)
        ax11 = fig.add_subplot(224)
        simulation = SimulationProducer(gui=True)
        simulation.add_background("ny_times_square")
        simulation.add_head()
        simulation.add_scale("only", (320, 320), 9.0)
        simulation.add_uniform_motion_screen("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/", size=1.5)
        simulation.start_sim()
        simulation.episode_reset_uniform_motion_screen(
            start_distance=2,
            depth_speed=0,
            angular_speed=1.125,
            direction=0,
            texture_id=0,
            preinit=True,
        )
        simulation.episode_reset_head(vergence=distance_to_vergence(2), cyclo=0)
        simulation.step_sim()
        simulation.step_sim()
        vision_0 = simulation.get_vision()
        simulation.step_sim()
        vision_1 = simulation.get_vision()
        im00 = ax00.imshow((0.5 + 0.5 * vision_0["only"][..., :3]))
        im10 = ax10.imshow((0.5 + 0.5 * vision_0["only"][..., 3:]))
        im01 = ax01.imshow((0.5 + 0.5 * vision_1["only"][..., :3]))
        im11 = ax11.imshow((0.5 + 0.5 * vision_1["only"][..., 3:]))
        plt.show()
        for i in range(50):
            simulation.episode_reset_uniform_motion_screen(
                start_distance=2,
                depth_speed=0,
                angular_speed=1.125,
                direction=0,
                texture_id=0,
                preinit=True,
            )
            for j in range(10):
                simulation.step_sim()


    def open_env():
        simulation = SimulationProducer(gui=True)
        simulation.start_sim()
        simulation.step_sim()
        simulation.add_background("ny_times_square")
        simulation.add_head()
        # simulation.add_scale("fine", (32, 32), 9.0)
        simulation.add_scale("coarse", (32, 32), 27.0)
        while True:
            simulation.step_sim()

    def demo():
        wait = 30
        speed = 0.1
        n = 100
        simulation = SimulationProducer(gui=True)
        simulation.start_sim()
        simulation.step_sim()
        simulation.add_background("ny_times_square")
        simulation.add_head()
        t_start = time.time()
        while time.time() < t_start + wait:
            simulation.step_sim()

        simulation.add_scale("only", (10, 10), 0.01)
        simulation.add_uniform_motion_screen("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/", size=1.5)
        simulation.episode_reset_uniform_motion_screen(
            start_distance=2,
            depth_speed=0,
            angular_speed=0,
            direction=0,
            texture_id=0,
            preinit=False,
        )

        # vergence
        for i in range(n):
            simulation.apply_action([0.0, 0.0, -speed, 0.0])
        for i in range(n):
            simulation.apply_action([0.0, 0.0, speed, 0.0])
        simulation.episode_reset_head(vergence=0, cyclo=0)

        # tilt
        simulation.apply_action([speed, 0.0, 0.0, 0.0])
        for i in range(n):
            # simulation.apply_action([0.0, 0.0, 0.0, 0.0])
            simulation.step_sim()
        simulation.apply_action([-2 * speed, 0.0, 0.0, 0.0])
        for i in range(n):
            # simulation.apply_action([0.0, 0.0, 0.0, 0.0])
            simulation.step_sim()
        simulation.episode_reset_head(vergence=0, cyclo=0)

        # pan
        simulation.apply_action([0.0, speed, 0.0, 0.0])
        for i in range(n):
            # simulation.apply_action([0.0, 0.0, 0.0, 0.0])
            simulation.step_sim()
        simulation.apply_action([0.0, -2 * speed, 0.0, 0.0])
        for i in range(n):
            # simulation.apply_action([0.0, 0.0, 0.0, 0.0])
            simulation.step_sim()
        simulation.episode_reset_head(vergence=0, cyclo=0)

        t_start = time.time()
        while time.time() < t_start + wait:
            simulation.step_sim()


    def test_accuracy():
        import matplotlib.pyplot as plt

        simulation = SimulationProducer(gui=True)
        simulation.start_sim()
        simulation.step_sim()
        simulation.add_background("ny_times_square")
        simulation.add_head()

        def test_vergence(speeds):
            speeds = np.array(speeds)
            positions = np.cumsum(speeds)
            recorded_positions = []
            recorded_speeds = []
            simulation.episode_reset_head(vergence=0, cyclo=0)
            for speed in speeds:
                simulation.apply_action([0.0, 0.0, speed, 0.0])
                recorded_positions.append(simulation.get_joints_positions()[2])
                recorded_speeds.append(simulation.get_joints_velocities()[2])
            simulation.episode_reset_head(vergence=0, cyclo=0)
            return positions, speeds, recorded_positions, recorded_speeds

        def display_results(positions, speeds, recorded_positions, recorded_speeds):
            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.plot(positions, label="target")
            ax.plot(recorded_positions, label="record")
            ax.legend()
            ax.set_title("position")
            ax = fig.add_subplot(122)
            ax.plot(speeds, label="target")
            ax.plot(recorded_speeds, label="record")
            ax.legend()
            ax.set_title("speed")
            plt.show()

        results = test_vergence([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        display_results(*results)
        results = test_vergence(np.random.uniform(size=20, low=-2, high=2))
        display_results(*results)

    def test_7():
        import matplotlib.pyplot as plt
        # plt.ion()
        fig = plt.figure()
        ax00 = fig.add_subplot(221)
        ax10 = fig.add_subplot(222)
        ax01 = fig.add_subplot(223)
        ax11 = fig.add_subplot(224)
        simulation = SimulationProducer(gui=True)
        simulation.add_background("ny_times_square")
        simulation.add_head()
        simulation.add_scale("only", (320, 320), 9.0)
        simulation.add_uniform_motion_screen("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/", size=1.5)
        simulation.start_sim()
        simulation.episode_reset_uniform_motion_screen(
            start_distance=2,
            depth_speed=0,
            angular_speed=1.125,
            direction=0,
            texture_id=0,
            preinit=True,
        )
        simulation.episode_reset_head(vergence=distance_to_vergence(2), cyclo=0)
        # simulation.step_sim()
        print(simulation.get_screen_position())
        # simulation.step_sim()
        vision_0 = simulation.get_vision()
        print(simulation.get_screen_position())
        simulation.step_sim()
        vision_1 = simulation.get_vision()
        print(simulation.get_screen_position())
        simulation.step_sim()
        vision_2 = simulation.get_vision()
        print(simulation.get_screen_position())
        simulation.step_sim()
        vision_3 = simulation.get_vision()
        print(simulation.get_screen_position())
        im00 = ax00.imshow((0.5 + 0.5 * vision_0["only"][..., :3]))
        im10 = ax10.imshow((0.5 + 0.5 * vision_1["only"][..., :3]))
        im01 = ax01.imshow((0.5 + 0.5 * vision_2["only"][..., :3]))
        im11 = ax11.imshow((0.5 + 0.5 * vision_3["only"][..., :3]))
        plt.show()
        # for i in range(50):
        #     simulation.episode_reset_uniform_motion_screen(
        #         start_distance=2,
        #         depth_speed=0,
        #         angular_speed=1.125,
        #         direction=0,
        #         texture_id=0,
        #         preinit=True,
        #     )
        #     for j in range(10):
        #         simulation.step_sim()

    class Scale:
        def __init__(self, resolution, view_angle):
            self.resolution = resolution
            self.view_angle = view_angle

    def test_fake_scales():
        import matplotlib.pyplot as plt

        simulation = SimulationProducer(gui=True)
        simulation.add_uniform_motion_screen("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/", size=1.5)
        simulation.start_sim()
        simulation.step_sim()
        simulation.add_background("ny_times_square")
        simulation.add_head()
        fine_scale = Scale(resolution=[32, 32], view_angle=9.0)
        coarse_scale = Scale(resolution=[32, 32], view_angle=27.0)
        scales = {
            "fine": fine_scale,
            "coarse": coarse_scale,
        }
        # simulation.add_scale("fine", (32, 32), 9.0)
        # simulation.add_scale("coarse", (32, 32), 27.0)
        simulation.add_fake_scales(scales)
        simulation.episode_reset_uniform_motion_screen(
            start_distance=2,
            depth_speed=0,
            angular_speed=0,
            direction=0,
            texture_id=0,
            preinit=False,
        )
        simulation.step_sim()
        simulation.step_sim()
        simulation.step_sim()
        simulation.step_sim()
        simulation.step_sim()
        simulation.step_sim()
        fake_vision = simulation.get_fake_vision()
        simulation.close()

        simulation = SimulationProducer(gui=True)
        simulation.add_uniform_motion_screen("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/", size=1.5)
        simulation.start_sim()
        simulation.step_sim()
        simulation.add_background("ny_times_square")
        simulation.add_head()
        fine_scale = Scale(resolution=[32, 32], view_angle=9.0)
        coarse_scale = Scale(resolution=[32, 32], view_angle=27.0)
        simulation.add_scale("fine", (32, 32), 9.0)
        simulation.add_scale("coarse", (32, 32), 27.0)
        simulation.episode_reset_uniform_motion_screen(
            start_distance=2,
            depth_speed=0,
            angular_speed=0,
            direction=0,
            texture_id=0,
            preinit=False,
        )
        simulation.step_sim()
        simulation.step_sim()
        simulation.step_sim()
        simulation.step_sim()
        simulation.step_sim()
        simulation.step_sim()
        vision = simulation.get_vision()
        simulation.close()

        fig = plt.figure()
        ax00 = fig.add_subplot(241)
        ax10 = fig.add_subplot(242)
        ax01 = fig.add_subplot(243)
        ax11 = fig.add_subplot(244)
        ax20 = fig.add_subplot(245)
        ax30 = fig.add_subplot(246)
        ax21 = fig.add_subplot(247)
        ax31 = fig.add_subplot(248)
        ax00.imshow((fake_vision["fine"][..., :3] + 1) * 2)
        ax10.imshow((fake_vision["fine"][..., 3:] + 1) * 2)
        ax01.imshow((fake_vision["coarse"][..., :3] + 1) * 2)
        ax11.imshow((fake_vision["coarse"][..., 3:] + 1) * 2)
        ax20.imshow((vision["fine"][..., :3] + 1) * 2)
        ax30.imshow((vision["fine"][..., 3:] + 1) * 2)
        ax21.imshow((vision["coarse"][..., :3] + 1) * 2)
        ax31.imshow((vision["coarse"][..., 3:] + 1) * 2)
        plt.show()

    test_fake_scales()
