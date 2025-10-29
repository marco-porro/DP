import gymnasium as gym   # ✅ cambiato da gym → gymnasium
import numpy as np
from diffusion_policy.real_world.video_recorder import VideoRecorder

class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            video_recoder: VideoRecorder,
            mode='rgb_array',
            file_path=None,
            steps_per_render=1,
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder

        self.step_count = 0

    def reset(self, **kwargs):
        # ✅ Gymnasium reset() → (obs, info)
        obs, info = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.video_recoder.stop()
        return obs, info   # ✅ return anche info

    def step(self, action):
        # ✅ Gymnasium step() → (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = super().step(action)
        self.step_count += 1
        if self.file_path is not None \
            and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)

            # ✅ Gymnasium: render() non accetta più il parametro "mode"
            frame = self.env.render(**self.render_kwargs)
            self.video_recoder.write_frame(frame)
            assert frame.dtype == np.uint8
            self.video_recoder.write_frame(frame)

        # ✅ manteniamo compatibilità: restituiamo i 5 elementi Gymnasium
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='rgb_array', **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path
