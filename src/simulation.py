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
            # write_coppeliasim_stdout_to_file=True
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
        self.scales_resolutions = {}

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

    def add_camera(self, eye, resolution, view_angle, downsampling):
        if self.head is None:
            raise ValueError("Can not add a camera with no head")
        else:
            position = self.head.get_eye_position(eye)
            orientation = self.head.get_eye_orientation(eye)
            vision_sensor = VisionSensor.create(
                resolution=[r * downsampling for r in resolution],
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
    def get_vision(self):
        return {
            scale_id: resize(
                np.concatenate([
                    self._cams[left].capture_rgb(),
                    self._cams[right].capture_rgb()
                    ], axis=-1) * 2 - 1,
                self.scales_resolutions[scale_id],
                anti_aliasing=True)
            for scale_id, (left, right) in self.scales.items()
        }

    @communicate_return_value
    def add_scale(self, id, resolution, view_angle, downsampling):
        if id in self.scales:
            raise ValueError("Scale with id {} is already present".format(id))
        else:
            self.head.set_joints_velocities(0, 0, 0, 0)
            self.head.set_joints_positions(0, 0, 0, 0)
            # self._pyrep.step() # step to make sure that the eyes reached the target position of 0 (might be useless)
            left = self.add_camera('left', resolution, view_angle, downsampling)
            right = self.add_camera('right', resolution, view_angle, downsampling)
            self.scales[id] = (left, right)
            self.scales_resolutions[id] = resolution[::-1]
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
            self.scales_resolutions.pop(id)
        else:
            raise ValueError("Scale with id {} does not exist".format(id))

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
    def test_install():
        simulation = SimulationProducer(gui=True)
        simulation.start_sim()
        simulation.step_sim()
        simulation.add_head()
        simulation.add_uniform_motion_screen("../textures/", size=1.5)
        for i in range(100):
            simulation.episode_reset_uniform_motion_screen()
            for j in range(20):
                print(i, end='\r')
                simulation.move_uniform_motion_screen()
                simulation.step_sim()
        simulation.stop_sim()

    test_install()
